FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    strace tcpdump curl ca-certificates \
    gcc libc6-dev \
    libssl-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*
COPY Geekbench-6.7.1-Linux /opt/geekbench
COPY geekbench.sh geekbench_capture.py /opt/
RUN chmod +x /opt/geekbench.sh
WORKDIR /opt
