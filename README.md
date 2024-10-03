<h1 align="center">RustFL: Secure and Asynchronous Federated Learning with Differential Privacy and Secure Multiparty Computation</h1>

This project demonstrates the implementation of a federated learning for deep learning in Rust. The client performs local training on the MNIST dataset using a SimpleCNN model, communicates with a server to fetch and send model weights, and aggregates models asynchronously. The server aggregates model weights and updates the global model. Additionally, the project includes advanced privacy and security features through Differential Privacy (DP) and Secure Multiparty Computation (SMPC).

## Release 1 description:

1. Created project directory hierarchy, Cargo.toml  for building the crate.
2. Developed the server code using Rust’s async libraries (e.g., tokio or async-std) for asynchronous client handling.
3. Implemented asynchronous handling of client connections, model update transmissions, and response handling.

## Crates Used
Here are the primary dependencies used in this project:

1. tch: Provides support for deep learning, PyTorch model implementation, and tensor operations.
2. ndarray: Provides support for N-dimensional arrays, used for processing data.
3. log: Provides logging capabilities.
4. reqwest: Used for making HTTP requests to the server for fetching and sending model weights.
5. tokio: Asynchronous runtime used to handle async tasks like HTTP requests.

## Project Flow for Release 1

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

                                       cargo run —bin client
3. Run the server code:

                                       cargo run —bin client

## Logging

The program uses the log crate to print informative messages, warnings, and errors during execution. The logs will help track the progress of each training round and any communication issues with the server.

## Next two Releases:

1. v2 (Month 2): Differential Privacy and Model Update Integration

    Objective: Add Differential Privacy (DP) to the model update process for client-side privacy.
    
    Tasks:

        1. Develop the module for DP to add noise to model updates before transmission.
        2. Ensure the DP mechanism is balanced, protecting privacy without sacrificing model accuracy.
        3. Securely transmit the differentially private model updates to the server.

2. v3 (Month 3): Secure Multiparty Computation (SMPC) and Aggregation

    Objective: Implement secure aggregation of model updates using SMPC.
    
    Tasks:
    
        1. Integrate cryptographic libraries (e.g., dalek-cryptography) to implement SMPC.
        2. Develop the aggregation logic to securely aggregate encrypted model updates using secret sharing techniques.
        3. Evaluate the security and performance of the aggregation process.
