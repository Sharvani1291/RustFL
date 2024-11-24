## Why Use RustFL for Federated Learning?

  The RustFL crate is specifically designed for privacy-preserving federated learning. Its key features include:
  
  Differential Privacy:
  
  Implements Gaussian noise mechanisms for ensuring privacy in model updates, making it suitable for applications with sensitive data.
  
  Secure Aggregation:
  
  Includes functionality for secret sharing and encryption to safeguard model weights during transmission and aggregation.
  
  Efficient Model Management:
  
  Supports federated averaging and provides tools to manage global models seamlessly in distributed environments.


## Running the Example

1. Build the Project:

                                                 cargo build

2. Run the Client:

                                                 cargo run --bin example_client

3. Run the Server:

                                                 cargo run --bin example_server
