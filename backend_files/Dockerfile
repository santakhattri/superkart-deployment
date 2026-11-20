FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install the dependencies first (this layer is cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the serialized model
COPY . .

# The Flask API listens on port 7860
EXPOSE 7860

# Serve the Flask app with gunicorn (production-grade WSGI server)
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
