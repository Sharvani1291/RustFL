use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use log::info;

// Logging setup
fn setup_logger() {
    env_logger::init();
}

// Global state for the model
#[derive(Clone)]
struct AppState {
    global_model: Mutex<SimpleCNN>,
    client_updates: Mutex<Vec<WeightsUpdate>>,
    current_model_version: Mutex<i32>,
    aggregation_goal: Mutex<i32>,
}

// Define the CNN model structure
#[derive(Debug)]
struct SimpleCNN {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl SimpleCNN {
    fn new(vs: &nn::Path) -> SimpleCNN {
        let conv1 = nn::conv2d(vs, 1, 32, 3, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 3, Default::default());
        let fc1 = nn::linear(vs, 64 * 7 * 7, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());

        SimpleCNN { conv1, conv2, fc1, fc2 }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 64 * 7 * 7])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

// Struct for deserializing the model weights update
#[derive(Deserialize, Serialize)]
struct WeightsUpdate {
    model_weights: Vec<Vec<f64>>,
    num_samples: i32,
    loss: f64,
    model_version: i32,
}

// Handler to get the current global model
#[get("/get_model")]
async fn get_model(data: web::Data<AppState>) -> impl Responder {
    let global_model = data.global_model.lock().unwrap();
    let model_state_dict = "Model State".to_string(); // Placeholder for model weights
    let current_model_version = *data.current_model_version.lock().unwrap();

    HttpResponse::Ok().json(serde_json::json!({
        "model_state_dict": model_state_dict,
        "model_version": current_model_version
    }))
}

// Handler to update the global model with client updates
#[post("/update_model")]
async fn update_model(
    weights: web::Json<WeightsUpdate>,
    data: web::Data<AppState>,
) -> impl Responder {
    let mut client_updates = data.client_updates.lock().unwrap();
    let aggregation_goal = *data.aggregation_goal.lock().unwrap();

    info!("Received model update from client with loss: {}", weights.loss);
    client_updates.push(weights.into_inner());

    if client_updates.len() >= aggregation_goal as usize {
        // Placeholder for model aggregation (FedAvg)
        client_updates.clear(); // Reset client updates after aggregation

        let mut current_model_version = data.current_model_version.lock().unwrap();
        *current_model_version += 1;

        HttpResponse::Ok().json(serde_json::json!({
            "message": "Global model updated",
            "model_version": *current_model_version
        }))
    } else {
        HttpResponse::Ok().json(serde_json::json!({
            "message": format!("Waiting for more client updates. Received {}/{}", client_updates.len(), aggregation_goal)
        }))
    }
}

// Main function to start the Actix web server
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    setup_logger();

    // Initialize the global model
    let vs = nn::VarStore::new(Device::Cpu);
    let global_model = SimpleCNN::new(&vs.root());

    let state = web::Data::new(AppState {
        global_model: Mutex::new(global_model),
        client_updates: Mutex::new(vec![]),
        current_model_version: Mutex::new(0),
        aggregation_goal: Mutex::new(1),  // Placeholder for aggregation goal
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(get_model)
            .service(update_model)
    })
    .bind(("127.0.0.1", 8081))?
    .run()
    .await
}
