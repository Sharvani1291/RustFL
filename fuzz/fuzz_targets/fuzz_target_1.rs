#![no_main]
use libfuzzer_sys::fuzz_target;
use RustFL::client::send_local_model_weights;
use RustFL::client::SimpleCNN;
use RustFL::server::nn::VarStore;
use RustFL::secure_dp_utils::generate_fernet_key;
use tch::Device;

fuzz_target!(|data: &[u8]| {
    let encryption_key = generate_fernet_key();
    let dummy_weights: Vec<f64> = data.iter().take(10).map(|&byte| byte as f64).collect();
    let dummy_loss = 0.1; // Example loss
    let dummy_version = 1;
    let post_url = "http://0.0.0.0:8081/update_model".to_string();
    let get_url = "http://0.0.0.0:8081/get_model".to_string();

    // Dummy SimpleCNN
    let device = Device::Cpu;
    let dummy_model = SimpleCNN::new(&VarStore::new(device).root());

    // Fuzz the send_local_model_weights function
    let _ = send_local_model_weights(
        dummy_weights,
        dummy_loss,
        dummy_version,
        &dummy_model,
        &encryption_key,
        device,
        &get_url,
        &post_url
    );
});
