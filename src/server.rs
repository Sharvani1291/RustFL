use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse};
use serde::{Deserialize, Serialize};
use log::info;
use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
struct WeightsUpdate {
    model_weights: Vec<String>,
    num_samples: usize,
    loss: f64,
    model_version: usize,
}

// Global state for model version and client updates
struct AppState {
    aggregation_goal: usize,
    vs: VarStore,
    current_model_version: Mutex<usize>,
    client_updates: Mutex<Vec<WeightsUpdate>>,
    global_model: Mutex<nn::Sequential>,
}

// Simple CNN using tch-rs (Rust bindings for PyTorch)
fn create_model(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 1, 32, 3, nn::ConvConfig::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.view([-1, 32 * 14 * 14]))
        .add(nn::linear(vs, 32 * 14 * 14, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 10, Default::default()))
}

// Federated averaging on encrypted weights (this example is simplified)
fn fed_avg_encrypted(weights_updates: Vec<Vec<String>>) -> Vec<String> {
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

async fn init_app_state() -> web::Data<AppState> {
    // Initialize VarStore and other fields as needed
    let vs = VarStore::new(tch::Device::Cpu);
    
    // Initialize your model and register it with vs
    let global_model = YourModelType::new(&vs.root()); // Make sure YourModelType is registered to vs

    web::Data::new(AppState {
        vs,
        global_model: Mutex::new(global_model),
        current_model_version: Mutex::new(1),
    })
}

#[get("/get_model")]
async fn get_model(data: web::Data<AppState>) -> impl Responder {
    let vs = &data.vs;
    let global_model = data.global_model.lock().unwrap();

    let model_state_dict = global_model
        .parameters()
        .iter()
        // .map(|(key, value)| (key.clone(), value.to_kind(tch::Kind::Float).to_vec()))
        .map(|(key, value)| {
            // converting tensor to Vec and handle nested structures
            let tensor_as_vec = match value.to_kind(tch::Kind::Float).view(-1).to_vec() {
                Ok(vec) => vec,
                Err(_) => {
                    // Handle any error in conversion and log for debugging
                    log::error!("Failed to convert tensor for key: {}", key);
                    vec![] // Or another fallback
                }
            };
            (key.clone(), tensor_as_vec)
        })
        .collect::<Vec<_>>();
        .collect::<Vec<_>>();

    HttpResponse::Ok().json(serde_json::json!({
        "model_state_dict": model_state_dict,
        "model_version": *data.current_model_version.lock().unwrap()
    }))
}

#[post("/update_model")]
async fn update_model(update: web::Json<WeightsUpdate>, data: web::Data<AppState>) -> impl Responder {
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

#[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init(); // Initialize logging

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let global_model = create_model(&vs.root());

    let state = web::Data::new(AppState {
        aggregation_goal: 1,
        current_model_version: Mutex::new(0),
        client_updates: Mutex::new(Vec::new()),
        global_model: Mutex::new(global_model),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(get_model)
            .service(update_model)
    })
    .bind(("0.0.0.0", 8081))?
    .run()
    .await
}
