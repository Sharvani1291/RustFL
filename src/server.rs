pub use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse};
pub use serde::{Deserialize, Serialize};
pub use log::info;
pub use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor};
pub use std::sync::{Arc, Mutex};
pub use reqwest::Response;
use crate::secure_dp_utils::fed_avg_encrypted;

//Implemented by Sharvani Chelumalla
/// Struct to represent weight updates sent to the server.
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsUpdate {
    pub model_weights: Vec<String>,
    pub num_samples: usize,
    pub loss: f64,
    pub model_version: usize,
}

//Implemented by Sai Pranavi Reddy Patlolla
/// Global state for model version and client updates
pub struct AppState {
    pub aggregation_goal: usize,
    pub current_model_version: Mutex<usize>,
    pub client_updates: Mutex<Vec<WeightsUpdate>>,
    pub global_model: Mutex<nn::Sequential>,
}
//Implemented by Sai Pranavi Reddy Patlolla
impl AppState{
    /// Default global state if not defined by user
    pub fn default() -> Self{
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let global_model = create_model(&vs.root());
        AppState {
            aggregation_goal: 1,
            current_model_version: Mutex::new(0),
            client_updates: Mutex::new(Vec::new()),
            global_model: Mutex::new(global_model)
        }
    }
}

//Implemented by Sharvani Chelumalla
/// A CNN construction using max-pooling and activation functions
pub fn create_model(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 1, 32, 3, nn::ConvConfig::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.view([-1, 32 * 14 * 14]))
        .add(nn::linear(vs, 32 * 14 * 14, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 10, Default::default()))
}

//Implemented by Sai Pranavi Reddy Patlolla
#[get("/get_model")]
/// Stores the global model weights such that client can fetch the global weights
pub async fn get_model(data: web::Data<AppState>) -> impl Responder {
    let global_model = data.global_model.lock().unwrap();
    /*let model_state_dict = global_model
        .parameters()
        .iter()
        .map(|(key, value)| (key.clone(), value.to_kind(tch::Kind::Float).to_vec()))
        .collect::<Vec<_>>();
     */
    let model_state_dict = "Model State Dict".to_string();

    HttpResponse::Ok().json(serde_json::json!({
        "model_state_dict": model_state_dict,
        "model_version": *data.current_model_version.lock().unwrap()
    }))
}

//Implemented by Sai Pranavi Reddy Patlolla
#[post("/update_model")]
/// Updates the global model each time client sends the updated version of weights
pub async fn update_model(update: web::Json<WeightsUpdate>, data: web::Data<AppState>) -> impl Responder {
    info!("Received model update from client with loss: {}",update.loss);

    let mut client_updates = data.client_updates.lock().unwrap();
    client_updates.push(update.into_inner());

    if client_updates.len() >= data.aggregation_goal {
        let selected_clients = client_updates.split_off(0); // Select clients for aggregation
        let encrypted_weights_list = selected_clients
            .iter()
            .map(|client| client.model_weights.clone())
            .collect::<Vec<_>>();

        let aggregated_encrypted_weights = fed_avg_encrypted(encrypted_weights_list);
        info!("Aggregation is successful!");

        let mut current_version = data.current_model_version.lock().unwrap();
        info!("Global model updated, Version: {}",current_version);
        *current_version += 1;

        HttpResponse::Ok().json(serde_json::json!({
            "message": "Global model updated with encrypted weights",
            "encrypted_model_weights": aggregated_encrypted_weights,
            "model_version": *current_version
        }))
    } else {
        HttpResponse::Ok().json(serde_json::json!({
            "message": format!(
                "Waiting for more client updates. Received {}/{} updates",
                client_updates.len(),
                data.aggregation_goal
            )
        }))
    }
}