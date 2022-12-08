# IR Benchmarking

Here we describe the steps for replicating results from the paper, [Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking](https://arxiv.org/abs/2212.01340).

### 1. Setting up Amazon Instance

To get started on an Amazon instance, please go to the [Amazon tutorial for EC2 linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html). There are a range of instances to choose from so please choose one with the appropriate memory and GPU resources for running the selected experiments. For example, for our most well-resourced instance, we used `p3.8xlarge`.

When initiating your instance, please select an Ubuntu machine image. This will be helpful for installing Docker and dependencies in later steps.

### 2. Installing Docker

To install Docker, please use the following [installation instructions](https://docs.docker.com/engine/install/ubuntu/). To run our experiments that use GPU, please download the [Docker Nvidia drivers](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu).

You can use the `Dockerfile` we include in the repository for running the experiments. Feel free to edit it as needed for testing different configurations and scripts.

### 3. Running experiments

To run an experiments, first initialize a Docker container using the Dockerfile and the correspondinng command:

```
sudo docker build -t pulkit/run_ir:1.0 .
```

Afterwards, perform the following command to run the selected configuration of `run_ir.py`:

```
sudo docker run -ti --name testing_2 pulkit/testing_2:1.0
```
