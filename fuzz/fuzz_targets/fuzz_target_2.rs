//fuzz_target_2.rs is contributed by Sharvani Chelumalla

use actix_web::{web, HttpResponse, App, HttpServer};
use libfuzzer_sys::fuzz_target; // Import the fuzz_target macro
use RustFL::server::{AppState, WeightsUpdate};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Set up the server
    HttpServer::new(|| {
        App::new()
            .app_data(web::Data::new(AppState::default()))
            .service(web::resource("/update_model").to(update_model)) // Define the update_model handler
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

// Fuzzing target
fuzz_target!(|data: &[u8]| {
    // Process the fuzz data
    // Dummy app state
    let app_state = AppState::default();
    let app_data = web::Data::new(app_state);
});

// The handler function for "/update_model"
async fn update_model(update: web::Json<WeightsUpdate>) -> HttpResponse {
    // Process the weights update and respond accordingly
    HttpResponse::Ok().json(update.into_inner()) // Just returning the data as a response for this example
}
