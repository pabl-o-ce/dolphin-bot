# ---------------------------------------
# Runtime environment
FROM nvidia/cuda:12.3.1-base-ubuntu22.04 as release

ENV TZ=Etc/UTC

WORKDIR /usr/src/app

COPY . /usr/src/app

ENV LLAMA_CUBLAS=1
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV CMAKE_ARGS="FORCE_CMAKE=1"

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential python3 python3-pip \
    && pip3 install --upgrade pip \
    && pip3 install \
    discord-py-interactions==5.11.0 \
    python-dotenv==1.0.0 \
    llama-cpp-python==0.2.27 \
    langchain==0.1.0 \
    langchain-community==0.0.12 \
    langchain-core==0.1.8 \
    redis==5.0.1 \
    openai==1.8.0 \
    langchain-openai==0.0.2.post1

CMD ["python3", "src/main.py"]