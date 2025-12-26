# Use official Ollama image as base
FROM ollama/ollama:latest

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your Streamlit app
COPY app.py .

# Pull your desired model at build time (change to your preferred model)
RUN ollama pull phi3:mini
# Optional: Add more models
# RUN ollama pull mistral:7b

# Expose ports: Ollama API + Streamlit
EXPOSE 11434
EXPOSE 7860

# Start Ollama server in background, then run Streamlit
CMD ollama serve & sleep 10 && streamlit run app.py --server.port=7860 --server.address=0.0.0.0