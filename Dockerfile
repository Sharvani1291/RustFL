# Use the official Rust image
FROM rust:latest

# Set the working directory inside the container
WORKDIR /usr/src/myapp

# Install dependencies for LibTorch (e.g., wget, unzip, and libclang)
#RUN apt-get update && apt-get install -y \
#    wget \
#    unzip \
#    libclang-dev \
#    build-essential

# Download and install LibTorch
#RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.11.0.zip -O /tmp/libtorch.zip && \
#    unzip /tmp/libtorch.zip -d /usr/local && \
#    rm /tmp/libtorch.zip

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
RUN pip3 install --break-system-packages torch==2.2.0 torchvision torchaudio

# Set the environment variable for using PyTorch
ENV LIBTORCH_USE_PYTORCH=1

# Set the LIBTORCH environment variable before running cargo build
#ENV LIBTORCH=/usr/local/libtorch
#ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
#COPY /Users/sai/Downloads/libtorch /usr/local/libtorch
ENV DYLD_LIBRARY_PATH=/Users/sai/Downloads/libtorch/lib
ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH

# Copy the current directory contents into the container
COPY . .

# Build the Rust application with release profile
RUN cargo build --release
#RUN cargo build

# Specify which binary to run (example_client or example_server)
# Uncomment the one you want as the default
#CMD ["./target/release/example_client"]
#CMD ["./target/release/example_server"]
