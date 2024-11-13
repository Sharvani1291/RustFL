use fernet::Fernet;
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Distribution};

pub struct DPMechanism {
    epsilon: f64,
    sensitivity: f64
}
impl DPMechanism {
    pub fn new(epsilon: f64, sensitivity: f64) -> DPMechanism {
        DPMechanism {
            epsilon,
            sensitivity
        }
    }
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

pub fn encrypt_share(share: &str, key: &str) -> Result<Vec<u8>, String> {
    // Create a Fernet instance from the provided key
    let fernet = Fernet::new(key).ok_or("Invalid Key");

    // Encrypt the share
    let encrypted_share = fernet?.encrypt(share.as_bytes());

    Ok(encrypted_share.into())
}

pub fn generate_fernet_key() -> String{
    Fernet::generate_key()
}