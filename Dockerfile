FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision with CUDA 11.8 support
RUN pip uninstall torch -y
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install additional packages for logging experiments
RUN pip install --no-cache-dir tqdm rich pyfiglet click

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Ensure log directory exists
RUN mkdir -p /app/logs

# Make the run script executable
RUN chmod +x /app/new_run.sh

# Run the script
ENTRYPOINT ["/app/new_run.sh"]
