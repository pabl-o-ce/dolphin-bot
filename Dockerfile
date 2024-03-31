# ---------------------------------------
# Runtime environment
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV TZ=Etc/UTC

WORKDIR /usr/src/app

COPY . /usr/src/app

# RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends dumb-init \
#     && apt-get install -y git build-essential python3 python3-pip

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Set environment variable
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
# ENV CUDA_DOCKER_ARCH=all

# Verify Python installation
# RUN python3 --version

# Install depencencies
RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install . --force-reinstall --only-binary=:all: --extra-index-url=https://smartappli.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121 --no-cache-dir
# RUN pip3 install llama-cpp-python>=0.2.56 --force-reinstall --only-binary=:all: --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121 --no-cache-dir
# RUN pip3 install discord-py-interactions>=5.11.0 python-dotenv>=1.0.0 langchain>=0.1.0 langchain-community>=0.0.12 langchain-core>=0.1.8 redis>=5.0.1 openai>=1.8.0 langchain-openai>=0.0.2.post1

CMD ["python3", "src/main.py"]
