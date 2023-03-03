# IR Benchmarking

Here we describe the steps for replicating results from the paper, [Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking](https://arxiv.org/abs/2212.01340).

### 1. Setting up Amazon Instance

To get started on an Amazon instance, please go to the [Amazon tutorial for EC2 linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html). There are a range of instances to choose from so please choose one with the appropriate memory and GPU resources for running the selected experiments. For example, for our most well-resourced instance, we used `p3.8xlarge`.

When initiating your instance, please select an Ubuntu machine image. This will be helpful for installing Docker and dependencies in later steps.

### 2. Installing Docker

To install Docker, please use the following [installation instructions](https://docs.docker.com/engine/install/ubuntu/). To run our experiments that use GPU, please download the [Docker Nvidia drivers](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu).

You can use the `Dockerfile` we include in the repository (found in `primeqa/examples/ir_benchmarking`) for running the experiments. Feel free to edit it as needed for testing different configurations and scripts.

### 3. Installing dependencies

Once you have Docker set up, please go to the [PrimeQA main page](https://github.com/primeqa/primeqa/tree/ir-benchmarking) and follow the instructions for installing the relevant dependencies and Java package.

After that, go to the folder `primeqa/examples/ir_benchmarking` and download the following three files from [this downloads folder](https://zenodo.org/record/7477643#.Y6nNGezMKdY): `psgs_w100.tsv`, `xorqa_dev_gmt.tsv`, and `msmarco.psg.kldR2.nway64.ib__colbert-400000`. These files are the document collection, queries, and zero-shot PLAID model checkpoint for XOR-TyDi, respectively.

### 4. Running experiments

To run experiments, first initialize a Docker container using the Dockerfile and the corresponding command:

```
sudo docker build -t ir_benchmark/run_ir:1.0 .
```

Afterwards, perform the following command to run the selected configuration of `run_ir.py`:

```
sudo docker run --rm --gpus 1 ir_benchmark/run_ir:1.0
```

To restrict memory, please see the following [Docker instructions on resource constraints](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu). For example, to restrict the previous command to 32 GB, you can run:


```
sudo docker run --rm --gpus 1 ir_benchmark/run_ir:1.0 --memory=32000m
```
