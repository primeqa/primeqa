
FROM nvidia/cuda:12.0.0-base-ubuntu22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
ENV CUDA_VISIBLE_DEVICES="0"

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /home/ubuntu/primeqa
COPY README.md ./
COPY VERSION ./
COPY experiment.sh ./
COPY jdk-11.0.1 ./
COPY setup.py ./
ADD . ./

RUN conda env create -f conda_env.yml
#RUN conda activate colbert
ENV PATH=/opt/conda/envs/colbert/bin:$PATH
SHELL ["conda", "run", "-n", "colbert", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-pip
RUN apt-get install git -y

CMD ["conda", "run", "--no-capture-output", "-n", "colbert", "sh", "experiment.sh"]



