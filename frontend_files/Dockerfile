FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install the dependencies first (this layer is cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . .

# Streamlit listens on port 8501
EXPOSE 8501

# The backend address is provided at run time with:  docker run -e BACKEND_URL=http://superkart-backend:7860 ...
# CORS / XSRF protection are switched off because the app is served behind the Codespaces port-forwarding proxy
# (otherwise file uploads can fail with a 403 error)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
