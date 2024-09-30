/******************************************************
use axum::{routing::get, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use tokio;

#[derive(Debug, Deserialize)]
struct WeightsUpdate {
    model_weights: Vec<Vec<f32>>,
    num_samples: usize,
    loss: f32,
    model_version: usize,
}

#[derive(Serialize)]
struct ModelStateResponse {
    model_state_dict: Vec<Vec<f32>>,
    model_version: usize,
}

struct GlobalModel {
    net: nn::Sequential,
    optimizer: nn::Optimizer,
    version: usize,
}

impl GlobalModel {
    fn new(vs: &nn::Path) -> Self {
        let net = nn::seq()
            .add(nn::conv2d(vs, 1, 32, 3, Default::default()))
            .add_fn(|x| x.max_pool2d_default(2))
            .add(nn::conv2d(vs, 32, 64, 3, Default::default()))
            .add_fn(|x| x.max_pool2d_default(2))
            .add_fn(|x| x.view([-1, 64 * 7 * 7]))
            .add(nn::linear(vs, 64 * 7 * 7, 128, Default::default()))
            .add(nn::linear(vs, 128, 10, Default::default()));

        let optimizer = nn::sgd(net.parameters(), 0.01, Default::default(), false);
        GlobalModel {
            net,
            optimizer,
            version: 0,
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.net.forward(input)
    }

    fn update_weights(&mut self, new_weights: Vec<Vec<f32>>) {
        // Implement FedAvg logic here: update model weights
    }
}

async fn get_model_handler(
    state: Arc<Mutex<GlobalModel>>,
) -> Json<ModelStateResponse> {
    let model = state.lock().unwrap();
    let model_weights: Vec<Vec<f32>> = model
        .net
        .parameters()
        .iter()
        .map(|p| p.view([-1]).to_vec())
        .collect();

    Json(ModelStateResponse {
        model_state_dict: model_weights,
        model_version: model.version,
    })
}

async fn update_model_handler(
    Json(weights_update): Json<WeightsUpdate>,
    state: Arc<Mutex<GlobalModel>>,
) -> Json<String> {
    let mut model = state.lock().unwrap();

    model.update_weights(weights_update.model_weights);
    model.version += 1;

    Json("Model updated".to_string())
}

#[tokio::main]
async fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let global_model = Arc::new(Mutex::new(GlobalModel::new(&vs.root())));

    let app = Router::new()
        .route("/get_model", get(get_model_handler))
        .route("/update_model", post(update_model_handler))
        .layer(axum::AddExtensionLayer::new(global_model));

    axum::Server::bind(&"0.0.0.0:8081".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
*****************************************************/
/*use warp::Filter;

#[tokio::main]
async fn main() {
    // Define a simple route
    let route = warp::path!("get_model")
        .map(|| warp::reply::json(&"This is the model response"));

    // Start the server
    warp::serve(route)
        .run(([0, 0, 0, 0], 8081))
        .await;
}*/

use warp::Filter;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct ModelUpdate {
    model_weights: Vec<f32>,
    loss: f32,
    model_version: i32,
}

#[tokio::main]
async fn main() {
    let update_route = warp::path("update_model")
        .and(warp::post())
        .and(warp::body::json())
        .map(|update: ModelUpdate| {
            println!("Received model update: {:?}", update);
            warp::reply::json(&"Model updated successfully")
        });

    // Start the server
    warp::serve(update_route)
        .run(([0, 0, 0, 0], 8081))
        .await;
}