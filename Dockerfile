FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Configure Streamlit for production
RUN mkdir -p ~/.streamlit && \
    echo "[server]\n\
headless = true\n\
port = 8501\n\
enableXsrfProtection = false\n\
enableCORS = false" > ~/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
