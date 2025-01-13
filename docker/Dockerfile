# Start from the PyTorch 1.10 CUDA 11.3 runtime image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set a working directory inside the container
WORKDIR /app

# Copy the entire project into /app (build context must include all files)
COPY . /app

# Upgrade pip if needed
RUN pip install --no-cache-dir --upgrade pip

# If you have other Python dependencies, you can install them here.
# E.g., if you have a requirements.txt at the root of your project:
RUN pip install --no-cache-dir -r docker/requirements.txt

# By default, start a bash session so you can manually run scripts
CMD ["/bin/bash"]
