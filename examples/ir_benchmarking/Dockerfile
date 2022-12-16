
FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

#RUN apt-get install nvidia-cuda-toolkit

# create env with python 3.9
RUN conda create -y -n primeqa python=3.9
#RUN conda activate primeqa
#ENV PATH=/opt/conda/envs/primeqa/bin:$PATH

WORKDIR /home/ubuntu/primeqa
#COPY requirements.txt ./
COPY README.md ./
COPY VERSION ./
COPY experiment.sh ./
#COPY primeqa/ir/run_ir.py ./
#COPY primeqa ./
COPY jdk-11.0.1 ./
COPY setup.py ./
#ADD primeqa ./
ADD . ./

RUN apt-get update && apt-get install -y \
    python-pip
RUN apt-get install git -y

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN cd /root/home
#RUN ls ./
RUN pip install -e .[all]

#RUN sh experiment.sh
#COPY testing_2.py ./

#CMD ["python", "testing_2.py"]
CMD ["sh", "experiment.sh"]



