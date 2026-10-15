# GPU image for DriveJudge: every environment the pipeline needs.
#
#   docker build -f docker/Dockerfile -t drivejudge .        # from the repo root
#   docker run --gpus all -it -v "$PWD":/workspace/DriveJudge \
#       -v <STORAGE>:<STORAGE> drivejudge
#
# The code is mounted, not copied: edit on the host, run in the container.
#
#   python                      open-source models, InternVL, vllm serve
#                               (requirements-gpu.txt + requirements-internvl.txt)
#   /opt/agent-venv/bin/python  the DriveJudge agent (requirements-agent.txt)

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1

# Python 3.12 as the default interpreter (the flash-attn wheel is cp312)
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl git ffmpeg && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100 && \
    python -m pip install --upgrade pip setuptools wheel && \
    apt-get purge -y software-properties-common && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/requirements
COPY requirements*.txt ./

# requirements-internvl.txt pulls in requirements-gpu.txt
RUN python -m pip install -r requirements-internvl.txt

RUN python -m venv /opt/agent-venv && \
    /opt/agent-venv/bin/pip install -r requirements-agent.txt

WORKDIR /workspace/DriveJudge
