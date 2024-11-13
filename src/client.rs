pub use std::fmt::Debug;
pub use log::{error, info, warn};
pub use reqwest::Client;
pub use serde_json::Value;
pub use tch::{kind, nn::{self, Conv2D, Linear, Module, Optimizer, OptimizerConfig, Sgd, VarStore}, Device, Kind, Tensor};
pub use serde::{Deserialize, Serialize};
use crate::secure_dp_utils::{DPMechanism,secret_share_weights,encrypt_share,generate_fernet_key};

// Struct to represent weight updates sent to the server.
#[derive(Serialize, Deserialize)]
pub struct WeightsUpdate {
    // Weights of the model.
    pub model_weights: Vec<String>,
    pub num_samples: usize,
    pub loss: f64,
    pub model_version: usize,
}

pub struct Config {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub noise_level: f64,
    pub num_rounds: usize,
    pub sensitivity: f64,
    pub epsilon: f64,
}

impl Config {
    pub fn default() -> Self {
        Config {
            learning_rate: 0.001,
            batch_size: 64,
            noise_level: 0.1,  // This can be adjusted for DP
            num_rounds: 3,
            sensitivity: 1.0,  // Sensitivity of the function (adjust as necessary)
            epsilon: 0.5,  // Privacy budget (adjust as necessary)
        }
    }
}

// Define the SimpleCNN model structure.
#[derive(Debug)]
pub struct SimpleCNN {
    conv1: Conv2D,
    //conv2: Conv2D,
    fc1: Linear,
    fc2: Linear,
}


impl SimpleCNN {
    pub fn new(vs: &nn::Path) -> SimpleCNN {
        let conv1 = nn::conv2d(vs, 1, 32, 3, Default::default());
        //let conv2 = nn::conv2d(vs, 32, 64, 3, Default::default());
        let fc1 = nn::linear(vs, 32 * 13 * 13, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());

        SimpleCNN {
            conv1,
            //conv2,
            fc1,
            fc2
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.view([-1, 1, 28, 28]); // Assuming batch size can vary
        //info!("Input shape: {:?}", xs.size());

        let xs = xs.apply(&self.conv1).relu();
        //info!("After conv1: {:?}", xs.size());

        let xs = xs.max_pool2d_default(2);
        //info!("After max_pool2d (1): {:?}", xs.size());

        // Update the view size based on Python output
        let xs = xs.view([-1, 32 * 13 * 13]);
        //info!("Flattened shape: {:?}", xs.size());

        let xs = xs.apply(&self.fc1).relu();
        //info!("After fc1: {:?}", xs.size());

        let xs = xs.apply(&self.fc2);
        //info!("After fc2: {:?}", xs.size());

        xs
    }
}

// Function to load and normalize training data.
pub fn get_train_data(data_dir: String) -> Vec<(Tensor, Tensor)> {
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

    impl Module for Normalize {
        fn forward(&self, input: &Tensor) -> Tensor {
            ((input.to_kind(Kind::Float) / 255.0) - &self.mean) / &self.stddev
        }
    }

    // Define normalization parameters.
    let mean = Tensor::from_slice(&[0.1307]).to_kind(Kind::Float);
    let stddev = Tensor::from_slice(&[0.3081]).to_kind(Kind::Float);
    let transform = Normalize::new(mean, stddev);

    // Load MNIST dataset.
    let dataset = tch::vision::mnist::load_dir(data_dir).unwrap();

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
        let label = label.view([-1]);
        train_dataset.push((image, label));
    }
    let mut batches = Vec::new();

    for chunk in train_dataset.chunks(Config::default().batch_size) {
        if chunk.is_empty() {
            continue; // Skip empty chunks
        }

        let batch_dataset_images = Tensor::cat(&chunk.iter().map(|(d, _)| d.copy()).collect::<Vec<_>>(), 0);
        let batch_dataset_labels = Tensor::cat(&chunk.iter().map(|(_, t)| t.copy()).collect::<Vec<_>>(), 0);

        // Log shapes
        //info!("Batch data shape: {:?}", batch_dataset_images.size());
        //info!("Batch target shape: {:?}", batch_dataset_labels.size());

        batches.push((batch_dataset_images, batch_dataset_labels));
    }

    batches
    //train_dataset
}

// Asynchronously start the training process.
pub async fn start_training(
    train_loader: Vec<(Tensor, Tensor)>,
    model: &mut SimpleCNN,
    optimizer: &mut Optimizer,
    criterion: &dyn Fn(&Tensor, &Tensor) -> Tensor,
    device: Device,
    get_url: &str,
) -> (f64, Vec<f64>,usize){
    let client = Client::new();

    // Fetch initial model version.
    let initial_response = client.get(get_url).send().await.unwrap();
    let data: Value = initial_response.json().await.unwrap();
    let model_version = data.get("model_version")
        .and_then(|v| v.as_f64())
        .map(|v| v as usize)
        .expect("model_version is not a valid Integer");
    // Training for a defined number of rounds.
    let mut loss_value= 0.0 ;
    let mut trained_weights= vec![];
    for round_num in 0..Config::default().num_rounds {
        info!("Round {}", round_num + 1);
        fetch_global_model(model,get_url).await.unwrap();

        // Train the local model and send weights to the server.
        let (avg_loss, train_weights) = train_local_model(&train_loader, model, optimizer, criterion, device);
        loss_value = avg_loss;
        trained_weights = train_weights
    }
    info!("Training completed for 3 rounds");
    (loss_value,trained_weights,model_version)

}

// Asynchronously fetch the global model from the server.
pub async fn fetch_global_model<'a>(model: &'a SimpleCNN,get_url: &str) -> Result<&'a SimpleCNN, reqwest::Error> {
    let client = Client::new();

    // Send GET request to fetch the global model.
    let response = client.get(get_url).send().await?;

    if response.status().is_success() {
        let data: Value = response.json().await?;

        let _global_model_weights = data.get("model_state_dict").unwrap();
        let model_version = data.get("model_version").cloned();

        // Load the fetched global model weights into the model.
        /*******************
         model.load_state_dict(global_model_weights, false)?; // Added in next release
        *********************/
        //model.load_state_dict(global_model_weights, false)?; // Added in next release
        info!("Fetched global model");
        //data.get("model_version")
        Ok(model)
    } else {
        error!("Failed to fetch global model");
        Err(reqwest::Error::from(response.error_for_status().unwrap_err()))
    }
}

// Function to train the local model.
pub fn train_local_model(
    train_loader: &Vec<(Tensor, Tensor)>,
    model: &mut SimpleCNN,
    optimizer: &mut Optimizer,
    criterion: &dyn Fn(&Tensor, &Tensor) -> Tensor,
    device: Device
) -> (f64, Vec<f64>) {
    //model.train();
    let mut running_loss = 0.0;
    info!("Training");

    for (batch_idx, (data, target)) in train_loader.iter().enumerate() {
        let data = data.to(device); //.view([-1,1,28,28]); ON HOLD
        //info!("Data shape: {:?}", data.size());
        let target = target.to(device);
        //info!("Target shape: {:?}", target.size());
        optimizer.zero_grad();

        /*println!("********************************");
        println!();
        println!("{}",data);
        println!();
        println!("********************************");
         */

        // Forward pass
        let output = model.forward(&data);
        //info!("Output shape: {:?}", output.size());
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
    /*******************
    (avg_loss, model.state_dict())
    *******************/
    (avg_loss, vec![1f64, 2f64]) // Added in Next Release

}

// Asynchronously send local model weights to the server.
pub async fn send_local_model_weights(
    weights: Vec<f64>,
    loss_value: f64,
    model_version: usize,
    model: &SimpleCNN,
    encryption_key: &str,
    _device: Device,
    get_url: &str,
    post_url: &str
) {
    let dp_mechanism = DPMechanism::new(Config::default().epsilon, Config::default().sensitivity);
    let model_weights_list_noisy: Vec<f64> = dp_mechanism.add_noise(&weights);
    let shared_weights = secret_share_weights(model_weights_list_noisy, 3, 2, Config::default().noise_level);
    let encrypted_shares = shared_weights
        .iter()
        .map(|share| encrypt_share(&format!("{:?}", share), encryption_key))
        .map(|encrypted_bytes| String::from_utf8(encrypted_bytes.unwrap()).unwrap())
        .collect();

    let client_updates = WeightsUpdate {
        model_weights: encrypted_shares,
        num_samples: weights.len() as usize,
        loss: loss_value as f64,
        model_version,
    };

    let client = Client::new();
    // Send the weight update as a JSON payload.
    let response = client.post(post_url)
        .json(&client_updates)
        .send()
        .await.unwrap();

    if response.status().is_success() {
        info!("Model update successful");
    } else if response.status().as_u16() == 409 {
        warn!("Model version mismatch. Fetching the latest model.");
        fetch_global_model(model,get_url).await.unwrap(); // Fetch the latest model if there's a version mismatch.
    } else {
        error!("Failed to send model update: {}", response.status());
    }
}