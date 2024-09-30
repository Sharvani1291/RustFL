use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::iter::Skip;
use std::ops::{Div, Sub};
use std::ptr::null;
use arrayfire::view;
use ndarray::{Array, Ix2, Ix4};
use numpy::ndarray;
//use ndarray::Array;
//use rand::thread_rng;
//use rand::Rng;
use log::{error, info, warn};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use reqwest::{Client, RequestBuilder};
use tch::{nn, nn::Conv2D, nn::ConvConfig, nn::Linear, nn::PaddingMode, nn::Module, nn::VarStore, Tensor, nn::Optimizer, Device, nn::OptimizerConfig, nn::Sgd, Kind, vision::dataset::Dataset, kind};
use neuro::layers::{MaxPool2D};
use serde_json::{json, Value};
use tokio::join;

//#[derive(Serialize, Deserialize)]
struct WeightUpdate {
    model_weights:VarStore,
    num_samples: i32,
    loss: f32,
    model_version: String,
}

#[derive(Debug)]
//#[derive(Serialize,Deserialize)]  // Derive both Serialize and Deserialize
struct SimpleCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    pool:  Tensor,
    fc1: Linear ,
    fc2: Linear ,
}

impl SimpleCNN {
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
        //let pool = *MaxPool2D::with_param((2, 2), (2, 2));
        let mp :Tensor;
        let pool = mp.max_pool2d(2, 2,0, Default::default(), Default::default());
        let fc1 = nn::linear(vs, 64 * 7 * 7, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        SimpleCNN {
            conv1,
            conv2,
            pool,
            fc1,
            fc2
        }
    }
}

impl nn::Module for SimpleCNN{
    fn forward(&self, xs:&Tensor) -> Tensor {
        let xs = xs.apply(&self.conv1)
                                .relu()
                                .max_pool2d(2,2,0,Default::default(),Default::default())
                                .apply(&self.conv2)
                                .relu()
                                .max_pool2d(2,2,0,Default::default(),Default::default())
                                .view([-1,64*7*7])
                                .apply(&self.fc1)
                                .relu()
                                .apply(&self.fc2);

        xs
    }
}

//async fn fetch_global_model(vs:&mut VarStore, model:&SimpleCNN) -> Result<SimpleCNN, reqwest::Error> {
async fn fetch_global_model(model:&SimpleCNN) -> Result<SimpleCNN, reqwest::Error> {
    let client = Client::new();
    let url = "http://0.0.0.0:8081/get_model";

    let response = client.get(url).send().await?;

    if response.status().is_success(){
        let data = response.json()?;
        //let global_model_weights = data["model_state_dict"];
        let global_model_weights = data.model_state_dict.into_iter().map(|(key,value)|(key,Tensor::from_slice(&value))).collect();
        
        model.load_state_dict(global_model_weights,false)?;
        info!("Fetched global model");
    }
    else{
        error!("Failed to fetch global model");
    }
    Ok(())
}

fn get_train_data() -> Vec<(Tensor,Tensor)>{
    let mean = Tensor::from_slice(&[0.1307]).to_kind(Kind::Float);
    let stddev = Tensor::from_slice(&[0.3081]).to_kind(Kind::Float);
    let transform = |x: &Tensor| -> Tensor {
        ((x.to_kind(Kind::Float) / 255.0) - mean) / stddev
    };
    let dataset = tch::vision::mnist::load_dir("mnist_data").unwrap();

    let train_dataset_images = dataset.train_images.apply(&transform);
    //let train_dataset_images = transform(dataset.train_images);
    let train_dataset_labels = dataset.train_labels.to_kind(kind::Kind::Int64);

    let subset_train_dataset_images = train_dataset_images.narrow(0,0,10000);
    let subset_train_dataset_labels = train_dataset_labels.narrow(0,0,10000);

    let train_dataset = subset_train_dataset_images
                        .into_iter()
                        .zip(subset_train_dataset_labels.into_iter())
                        .map(|(image,label)|(image.squeeze,label.squeeze()))
                        .collect();

    train_dataset

    //(subset_train_dataset_images,subset_train_dataset_labels)

}
fn train_local_model(train_loader:Vec<(Tensor,Tensor)>, model:&SimpleCNN, mut optimizer: Optimizer,
                     criterion:&dyn Fn(&Tensor,&Tensor) ->Tensor, device: Device) -> (f64, VarStore) {
    model.train();
    let mut running_loss = 0.0;
    info!("Training");

    for (batch_idx, (data, target)) in train_loader.iter().enumerate() {
        let data = data.to(device);
        let target = target.to(device);
        optimizer.zero_grad();

        let output = model.forward(&data);
        let loss = criterion(&output, &target);
        loss.backward();
        optimizer.step();

        running_loss += f64::from(loss);

        if batch_idx % 100 == 0 {
            info!("Batch {}/{}, Loss: {}",batch_idx,train_loader.len(),loss.double_value(&[]));
        }
    }
    let avg_loss = running_loss / train_loader.len() as f64;
        info!("Average Loss: {}", avg_loss);

    (avg_loss, model.var_store())
}

fn send_local_model_weights(weights:VarStore,loss_value:f64,model_version: String,model:&SimpleCNN, device: Device){

    let model_weight_lists = weights.variables()
                                                            .map(|tensor| tensor.flatten(0,-1)
                                                            .into_vec::<f64>()
                                                            .unwrap())
                                                            .collect();

    let weight_update = WeightUpdate{
        model_weights: model_weight_lists,
        num_samples: weights.len() as i32,
        loss: loss_value as f32,
        model_version,
    };

    let url = "http://0.0.0.0:8081/update_model";

    let client = Client::new();
    let response = client.post(url)
                    .json(&weight_update)
                    .send()?;

    if response.status().is_success() {
        info!("Model update successful");
        Ok(true)
    } else if response.status().as_u16() == 409 {
        warn!("Model version mismatch. Fetching the latest model.");
        //fetch_global_model(&mut VarStore::new(device),  model)?;
        fetch_global_model(model)?;
    } else {
        error!("Failed to send model update: {}", response.status());
        Ok(false)
    }
}


fn start_training(train_loader: Vec<(Tensor,Tensor)>,model:&SimpleCNN,optimizer: &Optimizer,criterion:&dyn Fn(&Tensor,&Tensor) ->Tensor,device: Device){
    let url = "http://0.0.0.0:8081/get_model";
    let client = Client::new();

    let initial_response = client.get(url).send()?.json();
    let model_version = initial_response["model_version"].as_i64().unwrap();

    for round_num in 0..3{
        info!("Round {}",round_num+1);
        //fetch_global_model(&mut VarStore::new(device),model);
        fetch_global_model(model);

        let (loss_value, trained_weights) = train_local_model(&train_loader,model,optimizer, criterion, device);
        send_local_model_weights(trained_weights,loss_value,model_version,model,device);
    }
    info!("Training completed for 3 rounds");
}

#[tokio::main]
async fn main() {
//fn main() {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let mut vs = VarStore::new(device);
    let simple_cnn_model = SimpleCNN::new(&vs.root());
    let criterion = simple_cnn_model.forward(&Default::default()).cross_entropy_for_logits(&Default::default());
    let mut optimizer = Sgd::default().build(&vs, 0.001).unwrap();

    //let input = Tensor::randn(&[1, 1, 28, 28], (tch::Kind::Float, device));
    //let target = Tensor::of_slice(&[0]).to_device(device); // Example target
    //let output = model.forward(&input);
    //let loss = output.log_softmax(-1, tch::Kind::Float).nll_loss(&target);

    //fetch_global_model(&mut vs, &simple_cnn_model);
    fetch_global_model(&simple_cnn_model).await;
    let train_loader = get_train_data();
    start_training(train_loader,&simple_cnn_model,&optimizer,&criterion,device)
}
