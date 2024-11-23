use actix_web::{web, HttpResponse, Responder};
use log::info;
use RustFL::secure_dp_utils::fed_avg_encrypted;
use RustFL::server::{get,  post, App, AppState, HttpServer, WeightsUpdate};

//Implemented by Sai Pranavi Reddy Patlolla

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

//Implemented by Sai Pranavi Reddy Patlolla
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

#[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init(); // Initialize logging

    HttpServer::new(move || {
        App::new()
            .app_data(AppState::default())
            .service(get_model)
            .service(update_model)
    })
        .bind(("0.0.0.0", 8081))?
        .run()
        .await
}

