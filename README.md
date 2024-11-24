<h1 align="center">RustFL: Secure and Asynchronous Federated Learning with Differential Privacy and Secure Multiparty Computation

## Project Overview

RustFL is a federated learning framework designed to securely and asynchronously train machine learning models while preserving the privacy of client data. By integrating Differential Privacy (DP) and Secure Multiparty Computation (SMPC), the framework ensures that sensitive information remains confidential throughout the training and aggregation process.

## Key Features

1. Privacy-Preserving: Differential Privacy is applied to each model update to obfuscate sensitive information from individual clients.
2. Secure Aggregation: The server performs model updates using encrypted shares, ensuring that no client data is exposed during the aggregation process.
3. Asynchronous Communication: The framework utilizes asynchronous communication between the clients and the server.

## Architecture

RustFL operates with a central server and multiple clients, where the clients perform local model training and then update the global model securely and privately.

1. Client Training and Model Update Generation

    Clients retrieve the latest global model from the server.
   
    Perform local training using their datasets.
   
    Generate shares of the updated model weights using a Secret Sharing Mechanism (Shamir’s Secret Sharing).
   
    Apply Differential Privacy to the shares to ensure that individual client data is not leaked.

3. Encryption and Transmission

    After applying differential privacy, the noisy shares are encrypted to ensure further privacy protection.
   
    The encrypted shares are sent to the server for aggregation.

4. Server Aggregation

    The server receives the encrypted noisy shares from the clients.
   
    Aggregates these shares using Secure Multiparty Computation (SMPC) techniques.
   
    Reconstructs the updated global model from the aggregated shares.
   
    Updates the global model with the new aggregated weights.

## Technologies Used

Rust: The main programming language for implementing the federated learning framework.

tokio: Asynchronous runtime for handling concurrent tasks efficiently.

tch (PyTorch bindings): Used to implement deep learning models and tensor operations in Rust.

reqwest: For making HTTP requests between clients and servers.

log: To log important information, warnings, and errors during the process.

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


## Running the System

1. Build the Project:

                                                           cargo build

2. Run the Client:

                                                           cargo run --bin client

3. Run the Server:

                                                           cargo run --bin server

## Logging:

You can enable detailed logging with:

                                                           RUST_LOG=info cargo run --bin client
                                                           

We have also developed an application which uses our crate: [https://github.com/Sharvani1291/RustFL/blob/main/Example/README.md]

Our crate can be downloaded from crates.io: [https://crates.io/crates/RustFL]

## Final Release Details

This is the final release of RustFL, where the framework has been fully implemented and optimized. It incorporates Differential Privacy for added privacy protection and Secure Multiparty Computation (SMPC) to ensure that model updates from clients are aggregated securely, maintaining the confidentiality of each client’s data.

The system is designed to be privacy-preserving, offering secure federated learning for decentralized machine learning applications.

## Conclusion

RustFL is a secure and efficient federated learning system that provides end-to-end privacy protection through Differential Privacy and Secure Multiparty Computation. By using a decentralized model training approach, this system ensures that sensitive client data remains private while enabling collaborative machine learning.
