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
