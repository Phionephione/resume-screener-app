# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- Install Tesseract and other system dependencies ---
# This is the magic step that solves all our previous problems.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell the server how to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]