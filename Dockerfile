#Dockerfile contributed by Sharvani Chelumalla and Sainath Talakanti

# Use the official Rust image
FROM rust:latest

# Set the working directory inside the container
WORKDIR /usr/src/myapp

# Install dependencies for LibTorch (e.g., wget, unzip, and libclang)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install LibTorch
#RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.0.zip -O libtorch.zip && \
#RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.0.zip -O libtorch.zip && \
#RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcu118.zip -O libtorch.zip && \
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.5.0%2Bcu118.zip -O libtorch.zip && \
    unzip libtorch.zip -d /usr/local/ && \
    rm libtorch.zip

# Install Python and pip3
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopenblas-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 --version && pip3 --version

# Download pytorch
#RUN pip3 install --upgrade pip
#RUN pip3 install --break-system-packages torch==2.2.0 torchvision torchaudio
RUN pip3 install --break-system-packages torch==2.5.0 torchvision torchaudio

RUN apt update && apt install -y g++
ENV CC=gcc
ENV CXX=g++


# Set the environment variable for using PyTorch
ENV LIBTORCH_USE_PYTORCH=1

# Set the LIBTORCH environment variable before running cargo build
#ENV LIBTORCH=/usr/local/libtorch
#COPY /Users/sai/Downloads/libtorch /usr/local/libtorch
#ENV DYLD_LIBRARY_PATH=RustFL/libtorch/lib
ENV DYLD_LIBRARY_PATH=/usr/local/libtorch/lib
ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib

# Copy the current directory contents into the container
COPY . .

# Build the Rust application with release profile
RUN cargo build --release
#RUN cargo build

# Specify which binary to run (example_client or example_server)
# Uncomment the one you want as the default
#CMD ["./target/release/example_client"]
#RUN cargo build --bin ex_server
#CMD ["./target/debug/ex_server"]
