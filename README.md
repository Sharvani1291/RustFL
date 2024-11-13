<h1 align="center">RustFL: Secure and Asynchronous Federated Learning with Differential Privacy and Secure Multiparty Computation</h1>

# Project Explanation

## Overview: 

This workflow involves a federated learning architecture where a server and multiple clients collaborate to train a machine learning model while preserving data privacy. The process incorporates Differential Privacy (DP) and Secure Multiparty Computation (SMPC) to ensure the confidentiality of client data and model updates.

1. Architecture Setup:

    The system comprises a central server and multiple client nodes. Each client operates independently, accessing a global model provided       by the server.
2. Client Training and Model Update Generation:

    Clients retrieve the global model from the server and perform local training using their respective datasets, such as the MNIST dataset.     Upon completing the training, each client generates shares of its updated model weights, utilizing a secret sharing mechanism.

3. Incorporating Differential Privacy:

    To enhance privacy, clients apply Differential Privacy techniques by adding calibrated noise to each share of the model weights. This        step ensures that the individual contributions of clients are obscured, thus protecting sensitive information during transmission.

4. Encryption and Transmission:

    After applying Differential Privacy, clients encrypt the noisy shares to further secure the model updates. The encrypted shares are then     transmitted to the central server for aggregation.
5. Server-Side Aggregation:

    Upon receiving the encrypted shares from all participating clients, the server performs aggregation on these shares rather than on the       original model weights. This aggregation process is designed to preserve privacy and prevent leakage of individual client information.

6. Model Reconstruction and Update:

    Once the server has aggregated the shares, it reconstructs the updated model weights. The global model is then updated with these new        weights, ensuring that the model benefits from the collective training efforts of all clients while maintaining the confidentiality of       their local datasets.

## Architecture Diagram:
![image](https://github.com/user-attachments/assets/c03ff1bc-2a81-42c2-a30c-7dcf61a46d3e)

## Release 1 description:

1. Created project directory hierarchy, Cargo.toml  for building the crate.
2. Developed the server code using Rust’s async libraries (e.g., tokio or async-std) for asynchronous client handling.
3. Implemented asynchronous handling of client connections, model update transmissions, and response handling.

## Crates Used
Here are the primary dependencies used in this project:

1. tch: Provides support for deep learning, PyTorch model implementation, and tensor operations.
2. log: Provides logging capabilities.
3. reqwest: Used for making HTTP requests to the server for fetching and sending model weights.
4. tokio: Asynchronous runtime used to handle async tasks like HTTP requests.

## Requirements

1. Arm architecture required

2. latest libtorch file need to be downloaded from official pytorch website

3. pytorch 2.5.0 version is required

4. for OpenSSL error: follow the steps

        Download openssl-3.4.0.tar.gz from github and extract it
        In the terminal, follow:
        cd /Absolute/path/to/openssl-3.4.0
        ./config --prefix=$HOME/openssl --openssldir=$HOME/openssl
        make
        make install
        export OPENSSL_DIR=$HOME/openssl
        export OPENSSL_LIB_DIR=$OPENSSL_DIR/lib
        export OPENSSL_INCLUDE_DIR=$OPENSSL_DIR/include
        export PKG_CONFIG_PATH=$OPENSSL_LIB_DIR/pkgconfig
        source ~/.bashrc
        ls $OPENSSL_LIB_DIR
        ls $OPENSSL_INCLUDE_DIR
        cd /Absolute/path/to/RustFL
        cargo clean
        cargo build

5. For torch not found error, follow following in terminal:

        python3 --version #verify the version and update
        python3 -m pip install --upgrade pip
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
        source /path/to/venv/bin/activate
        export LIBTORCH_USE_PYTORCH=1

6. For .dyld file not found error:

        export DYLD_LIBRARY_PATH=/Absolute/path/to/libtorch/lib

7. To run client code with logInfo:

        RUST_LOG=info cargo run --bin client

8. For Amd architecture:

    Pytorch 2.2.0 and tch 0.15.0 and appropriate libtorch file is required

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

                                       cargo run —bin server

## Logging

The program uses the log crate to print informative messages, warnings, and errors during execution. The logs will help track the progress of each training round and any communication issues with the server.

## Project Flow for Release 2

Sequence Overview

Client:

Train the local model.
Split the updated weights into shares using Shamir’s secret sharing mechanism.
Apply Differential Privacy (add noise) to each share.
Send noisy shares to different servers.
Apply Differential Privacy (add noise) to share.

Server:

Receive noisy shares from clients.
Perform aggregation using SMPC technique.
Reconstruct the global model from the aggregated shares.
Update the global model with new weights.
Implemented aggregation method for the encrypted weights.


## Project Flow for Release3

Established server and client connection.
Implemented transferring of encrypted noisy shares from client to server.
Federated Learning end-to-end flow is implemented.

Released our first version RustFL 0.1.0 in [link](crates.io)
[link](https://crates.io/crates/RustFL)


## Final Release tasks

Create crate for all the functionalities we have used.
Prepare the documentation.
