# Start from the PyTorch 1.10 CUDA 11.3 runtime image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# First copy requirements.txt and install dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==24.0.0 && \
    pip install --no-cache-dir uv==0.5.23 && \
    uv pip install -r /tmp/requirements.txt --system --no-cache --compile-bytecode

# Set a working directory inside the container
WORKDIR /app

# Copy the entire project into /app (build context must include all files)
COPY . /app

EXPOSE 8080

# By default, start a bash session so you can manually run scripts
CMD ["/bin/bash"]
