use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse};
use serde::{Deserialize, Serialize};
use log::info;
use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor};
use std::sync::{Arc, Mutex};
use reqwest::Response;

#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsUpdate {
    model_weights: Vec<String>,
    num_samples: usize,
    loss: f64,
    model_version: usize,
}

// Global state for model version and client updates
pub struct AppState {
    aggregation_goal: usize,
    current_model_version: Mutex<usize>,
    client_updates: Mutex<Vec<WeightsUpdate>>,
    global_model: Mutex<nn::Sequential>,
}

// Simple CNN using tch-rs (Rust bindings for PyTorch)
pub fn create_model(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 1, 32, 3, nn::ConvConfig::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.view([-1, 32 * 14 * 14]))
        .add(nn::linear(vs, 32 * 14 * 14, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 10, Default::default()))
}

#[get("/get_model")]
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

#[post("/update_model")]
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

// Federated averaging on encrypted weights (this example is simplified)
pub fn fed_avg_encrypted(weights_updates: Vec<Vec<String>>) -> Vec<String> {
    let mut aggregated_weights = Vec::new();

    // Perform simple aggregation (just for demonstration)
    for i in 0..weights_updates[0].len() {
        let encrypted_sum = weights_updates
            .iter()
            .fold(weights_updates[0][i].clone(), |sum, client_weights| {
                sum + &client_weights[i] // Simplified string concatenation
            });
        aggregated_weights.push(encrypted_sum);
    }

    aggregated_weights
}