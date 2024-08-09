# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip freeze > /tmp/installed_requirements.txt && \
    echo "Installed packages:" && \
    cat /tmp/installed_requirements.txt

# Install PyTorch with CUDA 11.8
RUN pip uninstall torch -y && \
    pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run train.py when the container launches
CMD ["python", "train.py"]
