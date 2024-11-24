use fernet::Fernet;
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Distribution};

//Implemented by Sharvani Chelumalla
/// Structure for noise parameters
pub struct DPMechanism {
    epsilon: f64,
    sensitivity: f64
}
//Implemented by Sharvani Chelumalla
impl DPMechanism {
    /// Takes the default parameters or the ones defined by user
    pub fn new(epsilon: f64, sensitivity: f64) -> DPMechanism {
        DPMechanism {
            epsilon,
            sensitivity
        }
    }
    //Implemented by Sharvani Chelumalla
    /// Add noise to weights for privacy concerns
    pub fn add_noise(&self, weights: &Vec<f64>) -> Vec<f64> {
        let noise_std = self.sensitivity / self.epsilon;
        let normal_dist = Normal::new(0.0, noise_std).unwrap();
        let mut rng = thread_rng();

        // Adding Gaussian noise to each weight
        weights
            .iter()
            .map(|&weight| weight + normal_dist.sample(&mut rng))
            .collect()
    }
}

//Implemented by Sharvani Chelumalla
/// To add extra noise such that weights can be shared secretly
pub fn secret_share_weights(weights: Vec<f64>, num_shares: usize, threshold: usize, _noise_level: f64) -> Vec<Vec<f64>> {
    // Create a vector of vectors to hold shares for each shareholder
    let mut shares = vec![vec![]; num_shares];
    let mut rng = thread_rng();

    for &weight in &weights {
        // Generate random coefficients for the polynomial of degree (threshold - 1)
        let mut coeffs: Vec<f64> = (0..(threshold - 1)).map(|_| rng.gen_range(0.0..100.0)).collect();  // Generate random coefficients

        // Include the secret as the constant term of the polynomial
        coeffs.insert(0, weight);

        for i in 0..num_shares {
            // Calculate the share by evaluating the polynomial at (i + 1)
            let x = (i + 1) as f64;
            let share: f64 = coeffs.iter()
                .enumerate()
                //.map(|(idx, &coeff)| coeff * x.powi(idx as i32))  // x^idx * coeff
                //.sum();
                .fold(0.0, |acc, (idx, &coeff)| acc + coeff * (x.powi(idx as i32)));
            shares[i].push(share);  // Append the share to the shares vector
        }
    }

    shares  // Return the shares as a vector of vectors
}

//Implemented by Sainath Talaknati
/// Encrypt the weights using Fernet encryption key
pub fn encrypt_share(share: &str, key: &str) -> Result<Vec<u8>, String> {
    // Create a Fernet instance from the provided key
    let fernet = Fernet::new(key).ok_or("Invalid Key");

    // Encrypt the share
    let encrypted_share = fernet?.encrypt(share.as_bytes());

    Ok(encrypted_share.into())
}

//Implemented by Sainath Talaknati
/// Generates Encryption key using Fernet
pub fn generate_fernet_key() -> String{
    Fernet::generate_key()
}

//Implemented by Sai Pranavi Reddy Patlolla
/// Aggregates the received encrypted weights with the global model weights
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

//Tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use rand_distr::{Normal, Distribution};

    // Test for DPMechanism: add_noise function
    #[test]
    fn test_add_noise() {
        let dp_mechanism = DPMechanism::new(1.0, 1.0);
        let weights = vec![1.0, 2.0, 3.0];

        let noisy_weights = dp_mechanism.add_noise(&weights);

        assert_eq!(noisy_weights.len(), weights.len());
        for (original, noisy) in weights.iter().zip(noisy_weights.iter()) {
            // Check that noise has been added
            assert!((*original - *noisy).abs() > 0.0);
        }
    }

    // Test for secret_share_weights function
    #[test]
    fn test_secret_share_weights() {
        let weights = vec![10.0, 20.0, 30.0];
        let num_shares = 1;
        let threshold = 1; // Example threshold

        let shares = secret_share_weights(weights.clone(), num_shares, threshold, 0.0);

        // Check if the number of shares matches the num_shares input
        assert_eq!(shares.len(), num_shares);
        assert_eq!(shares[0].len(), weights.len());
    }

    // Test for encrypt_share function
    #[test]
    fn test_encrypt_share() {
        let share = "share_data";
        let key = generate_fernet_key();

        match encrypt_share(share, &key) {
            Ok(encrypted_share) => {
                assert!(!encrypted_share.is_empty());
            }
            Err(_) => panic!("Encryption failed"),
        }
    }

    // Test for generate_fernet_key function
    #[test]
    fn test_generate_fernet_key() {
        let key = generate_fernet_key();
        assert_eq!(key.len(), 44); // Fernet keys are always 44 bytes in base64 encoding
    }

    // Test for fed_avg_encrypted function
    #[test]
    fn test_fed_avg_encrypted() {
        let client_updates = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["d".to_string(), "e".to_string(), "f".to_string()],
        ];

        let aggregated_weights = fed_avg_encrypted(client_updates);

        // Check if the aggregation result is correct
        assert_eq!(aggregated_weights.len(), 3); // Should match the number of weights
    }
}