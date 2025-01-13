# Docker Instructions

This Docker setup starts from the PyTorch 1.10.0 CUDA 11.3 runtime image, which includes:
- Python 3.8 (as provided by the base image)
- PyTorch 1.10.0
- CUDA 11.3 and cuDNN 8 support

## Building the Image

From the **root directory** of your project (where `main.py` and the `docker` folder reside), run:

```bash
docker build -f docker/Dockerfile -t anomaly_transformer .
```

This uses the Dockerfile located in the docker folder and tags the built image as anomaly_transformer.
# Running the Container
After building, run the container interactively.
- **GPU Support**: Ensure that your host system has the appropriate NVIDIA drivers and Docker GPU support set up (e.g., nvidia-container-toolkit) so that the container can access the GPU. You can run:

```bash
docker run --gpus all -it anomaly_transformer
```

This command will drop you into a Bash shell `(/bin/bash)` inside the container with all the project files copied to `/app`.

Currently, if no NVIDIA GPU is present, the code des not work.

## Usage

1. Navigate to the scripts folder:

```bash
cd scripts
```

2. Run any script you like, for example:
```bash 
bash SMAP.sh
```

3. Exit the container when you are finished:

```bash
exit
```

# Notes and Tips

- **Additional Dependencies**: If your scripts require additional Python packages, add them to a `requirements.txt` file in the project root (or modify the Dockerfile to install them directly).

- **Volume Mounting**: If you prefer to run your project code outside of the container image, you can mount the local directory:

```bash
docker run --gpus all -it -v $(pwd):/app anomaly_transformer
```

This way, any changes you make locally are reflected in the container.

- **Maintaining Data**: If your scripts produce large outputs (e.g., logs, models), consider mounting a separate volume or directory on the host for persistent storage

