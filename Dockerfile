# Use an official Python runtime as a parent image
FROM python:3.7.16-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (if any)
COPY . .

# Command to run your training script
CMD ["python", "main.py"]
