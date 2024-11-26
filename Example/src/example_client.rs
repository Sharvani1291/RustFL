use RustFL::client::{WeightsUpdate, Config, SimpleCNN, Client, Value,
                     get_train_data, start_training, kind,
                     nn::{self, Conv2D, Linear, Module, Optimizer, OptimizerConfig, Sgd, VarStore},
                     Device, Kind, Tensor,error, info, warn,Deserialize, Serialize, Debug};
use RustFL::secure_dp_utils::{DPMechanism,generate_fernet_key,secret_share_weights,encrypt_share};

//Client example is contributed by Sainath Talaknati & Sharvani Chelumalla
// Main function to initialize and start the training process.
#[tokio::main]
async fn main() {
    env_logger::init();


    let config = Config::new(0.5, 128, 0.5, 5, 0.5, 1.5);

    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let vs = VarStore::new(device);
    let mut simple_cnn_model = SimpleCNN::new(&vs.root());
    let mut optimizer = Sgd::default().build(&vs, config.learning_rate).unwrap();

    // Define the loss function.
    let criterion = |output: &Tensor, target: &Tensor| {
        output
            .cross_entropy_for_logits(target)
            .mean(Kind::Float)
    };

    // Load the training data.
    let train_loader = get_train_data("mnist_data/MNIST/raw".to_string());

    let encryption_key = generate_fernet_key();

    let get_url = "http://0.0.0.0:8081/get_model";
    let post_url = "http://0.0.0.0:8081/update_model";
    let (loss_value,trained_weights,model_version) = start_training(train_loader,&mut simple_cnn_model, &mut optimizer, &criterion, device,get_url).await;
    RustFL::client::send_local_model_weights(trained_weights, loss_value, model_version.clone(), &mut simple_cnn_model, encryption_key.as_str(), device, get_url, post_url).await;
    info!("Model training has been completed.");
}