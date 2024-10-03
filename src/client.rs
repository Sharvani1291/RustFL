use std::fmt::Debug;
use log::{error, info, warn};
use reqwest::Client;
use serde_json::Value;
use tch::{kind, nn::{self, Conv2D, Linear, Module, Optimizer, OptimizerConfig, Sgd, VarStore}, Device, Kind, Tensor};
use serde::{Deserialize, Serialize};

// Struct to represent weight updates sent to the server.
#[derive(Serialize, Deserialize)]
struct WeightUpdate {
    // Weights of the model.
    model_weights: Vec<Vec<f64>>,
    num_samples: i32,
    loss: f32,
    model_version: String,
}

// Define the SimpleCNN model structure.
#[derive(Debug)]
struct SimpleCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    fc1: Linear,
    fc2: Linear,
}

// Implementation of the SimpleCNN model.
impl SimpleCNN {
    // Create a new instance of SimpleCNN.
    fn new(vs: &nn::Path) -> SimpleCNN {
        let conv1 = nn::conv2d(vs, 1, 32, 3, nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        });
        let conv2 = nn::conv2d(vs, 32, 64, 3, nn::ConvConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        });
        let fc1 = nn::linear(vs, 64 * 7 * 7, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        SimpleCNN {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

// Implementation of the nn::Module trait for SimpleCNN.
impl nn::Module for SimpleCNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.conv1)
            .relu()
            .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
            .apply(&self.conv2)
            .relu()
            .max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
            .view([-1, 64 * 7 * 7])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2);

        xs
    }
}

// Asynchronously fetch the global model from the server.
async fn fetch_global_model(model: &SimpleCNN) -> Result<SimpleCNN, reqwest::Error> {
    let client = Client::new();
    let url = "http://0.0.0.0:8081/get_model";

    // Send GET request to fetch the global model.
    let response = client.get(url).send().await?;

    if response.status().is_success() {
        let data: Value = response.json().await?;
        let global_model_weights = data["model_state_dict"];
        
        // Load the fetched global model weights into the model.
        model.load_state_dict(global_model_weights, false)?;
        info!("Fetched global model");
    } else {
        error!("Failed to fetch global model");
    }
    Ok(*model)
}

// Function to load and normalize training data.
fn get_train_data() -> Vec<(Tensor, Tensor)> {
    #[derive(Debug)]
    struct Normalize {
        mean: Tensor,
        stddev: Tensor,
    }

    impl Normalize {
        fn new(mean: Tensor, stddev: Tensor) -> Self {
            Normalize { mean, stddev }
        }
    }

    impl nn::Module for Normalize {
        fn forward(&self, input: &Tensor) -> Tensor {
            ((input.to_kind(Kind::Float) / 255.0) - &self.mean) / &self.stddev
        }
    }

    // Define normalization parameters.
    let mean = Tensor::from_slice(&[0.1307]).to_kind(Kind::Float);
    let stddev = Tensor::from_slice(&[0.3081]).to_kind(Kind::Float);
    let transform = Normalize::new(mean, stddev);

    // Load MNIST dataset.
    let dataset = tch::vision::mnist::load_dir("mnist_data").unwrap();

    // Normalize and subset training dataset.
    let train_dataset_images = transform.forward(&dataset.train_images);
    let train_dataset_labels = dataset.train_labels.to_kind(kind::Kind::Int64);
    let subset_train_dataset_images = train_dataset_images.narrow(0, 0, 10000);
    let subset_train_dataset_labels = train_dataset_labels.narrow(0, 0, 10000);

    // Create a vector of tuples (image, label) for training data.
    let mut train_dataset = Vec::new();
    for i in 0..subset_train_dataset_images.size()[0] {
        let image = subset_train_dataset_images.get(i).squeeze();
        let label = subset_train_dataset_labels.get(i).squeeze();
        train_dataset.push((image, label));
    }

    train_dataset
}

// Function to train the local model.
fn train_local_model(
    train_loader: &Vec<(Tensor, Tensor)>, 
    model: &mut SimpleCNN, 
    optimizer: &mut Optimizer, 
    criterion: &dyn Fn(&Tensor, &Tensor) -> Tensor, 
    device: Device
) -> (f64, VarStore) {
    model.train();
    let mut running_loss = 0.0;
    info!("Training");

    for (batch_idx, (data, target)) in train_loader.iter().enumerate() {
        let data = data.to(device);
        let target = target.to(device);
        optimizer.zero_grad();

        // Forward pass
        let output = model.forward(&data);
        let loss = criterion(&output, &target);
        loss.backward();
        optimizer.step();

        running_loss += loss.double_value(&[]);

        // Log the loss every 100 batches.
        if batch_idx % 100 == 0 {
            info!("Batch {}/{}, Loss: {}", batch_idx, train_loader.len(), loss.double_value(&[]));
        }
    }

    let avg_loss = running_loss / train_loader.len() as f64;
    info!("Average Loss: {}", avg_loss);

    (avg_loss, model.var_store())
}

// Asynchronously send local model weights to the server.
async fn send_local_model_weights(
    weights: VarStore, 
    loss_value: f64, 
    model_version: String, 
    model: &SimpleCNN, 
    device: Device
) {
    let model_weight_lists = weights.variables().into_iter()
        .map(|(_name, tensor)| {
            tensor.flatten(0, -1)
            .to_kind(Kind::Float)
            .try_into()
            .unwrap()
        })
        .collect();

    let weight_update = WeightUpdate {
        model_weights: model_weight_lists,
        num_samples: weights.len() as i32,
        loss: loss_value as f32,
        model_version,
    };

    let url = "http://0.0.0.0:8081/update_model";

    let client = Client::new();
    // Send the weight update as a JSON payload.
    let response = client.post(url)
        .json(&weight_update)
        .send()
        .await.unwrap();

    if response.status().is_success() {
        info!("Model update successful");
    } else if response.status().as_u16() == 409 {
        warn!("Model version mismatch. Fetching the latest model.");
        fetch_global_model(model).await.unwrap(); // Fetch the latest model if there's a version mismatch.
    } else {
        error!("Failed to send model update: {}", response.status());
    }
}

// Asynchronously start the training process.
async fn start_training(
    train_loader: Vec<(Tensor, Tensor)>, 
    model: &mut SimpleCNN, 
    optimizer: &mut Optimizer, 
    criterion: &dyn Fn(&Tensor, &Tensor) -> Tensor, 
    device: Device
) {
    let url = "http://0.0.0.0:8081/get_model";
    let client = Client::new();

    // Fetch initial model version.
    let initial_response = client.get(url).send().await.unwrap();
    let data: Value = initial_response.json().await.unwrap();
    let model_version = data["model_version"].as_str().unwrap().to_string();

    // Training for a defined number of rounds.
    for round_num in 0..3 {
        info!("Round {}", round_num + 1);
        fetch_global_model(model).await.unwrap();

        // Train the local model and send weights to the server.
        let (loss_value, trained_weights) = train_local_model(&train_loader, model, optimizer, criterion, device);
        send_local_model_weights(trained_weights, loss_value, model_version.clone(), model, device).await;
    }
    info!("Training completed for 3 rounds");
}

// Main function to initialize and start the training process.
#[tokio::main]
async fn main() {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let vs = VarStore::new(device);
    let mut simple_cnn_model = SimpleCNN::new(&vs.root());
    let mut optimizer = Sgd::default().build(&vs, 0.001).unwrap();

        // Define the loss function.
        let criterion = |output: &Tensor, target: &Tensor| {
            output
                .cross_entropy_for_logits(target)
                .mean(Kind::Float)
        };
    
        // Load the training data.
        let train_loader = get_train_data();
    
        // Start the training process asynchronously.
        start_training(train_loader, &mut simple_cnn_model, &mut optimizer, &criterion, device).await;
    
        info!("Model training has been completed.");
    }
    
