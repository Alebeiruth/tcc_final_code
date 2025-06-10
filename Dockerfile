FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY notebooks/ ./notebooks/
COPY final.py .
COPY README.md .

# Criar diretório para resultados
RUN mkdir -p results

# Comando padrão: Jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]