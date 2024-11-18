use RustFL::server::{get_model, update_model, App, AppState, HttpServer};

//Implemented by Sai Pranavi Reddy Patlolla
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

