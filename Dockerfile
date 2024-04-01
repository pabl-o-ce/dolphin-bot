# ---------------------------------------
# Runtime environment
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV TZ=Etc/UTC

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3.11 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Set environment variable
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
# ENV CUDA_DOCKER_ARCH=all

# Install depencencies
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install . --force-reinstall --only-binary=:all: --extra-index-url=https://smartappli.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122 --no-cache-dir

CMD ["python3", "src/main.py"]
