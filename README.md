<h1 align="center">RustFL: Secure and Asynchronous Federated Learning with Differential Privacy and Secure Multiparty Computation</h1>

This project demonstrates the implementation of a federated learning for deep learning in Rust. The client performs local training on the MNIST dataset using a SimpleCNN model, communicates with a server to fetch and send model weights, and aggregates models asynchronously. The server aggregates model weights and updates the global model.

## Crates Used
Here are the primary dependencies used in this project:

1. tch: Provides support for deep learning, PyTorch model implementation, and tensor operations.
2. ndarray: Provides support for N-dimensional arrays, used for processing data.
3. log: Provides logging capabilities.
4. reqwest: Used for making HTTP requests to the server for fetching and sending model weights.
5. tokio: Asynchronous runtime used to handle async tasks like HTTP requests.

The Cargo.toml file should include these dependencies:

                            [dependencies]
                            #Client-only
                            reqwest = { version = "0.12.7", features = ["json"] }
                            ndarray = { version = "0.16.1",features = ["serde"]}
                            numpy = "0.21.0"
                            ndarray-rand = "0.15"
                            ndarray-npy = "0.9.1"
                            
                            #Client-Server
                            tch = "0.8.0"
                            #torch-sys = "0.17.0"
                            #tch-serde = "0.8.0"
                            serde = { version = "1.0", features = ["derive"] }
                            serde_json = "1.0"
                            validator = "0.18.1"
                            log = "0.4"
                            env_logger = "0.11.5"
                            
                            #Server-only
                            tokio = { version = "1", features = ["full"] }
                            axum = "0.7.6"

## Project Flow

1. Model Definition: The SimpleCNN struct defines the architecture of a convolutional neural network with two convolutional layers, a max-pooling layer, and two fully connected layers.
2. Local Training: The train_local_model function uses the SimpleCNN model to train on a subset of the MNIST dataset, applying stochastic gradient descent (SGD) as the optimizer.
3. Global Model Fetching: The client communicates with a server using reqwest to fetch the latest global model.
4. Local Model Weight Update: After local training, the client sends the updated model weights, number of samples trained, and loss values to the server via an HTTP POST request.
5. Server Aggregation: The server aggregates weights received from multiple clients and updates the global model.

## How to Run

Prerequisites

1. Rust and Cargo: Ensure that Rust and Cargo are installed on your system. You can install Rust from here.
2. PyTorch: The tch crate requires PyTorch's libtorch library. Install it from the official PyTorch website or follow the tch-rs documentation.

## Steps to Run the Client
1. Install the dependencies:

                                       cargo build
2. Run the client code:

                                       cargo run
3. Run the server code:

                                       cargo run

## Logging

The program uses the log crate to print informative messages, warnings, and errors during execution. The logs will help track the progress of each training round and any communication issues with the server.

## Conclusion

This Rust program implements a federated learning client that communicates asynchronously with a server. It uses the SimpleCNN architecture to perform local training on the MNIST dataset and supports model aggregation through HTTP requests. The client-server interaction follows the basic principles of federated learning, allowing for scalable model training across distributed edge devices.
