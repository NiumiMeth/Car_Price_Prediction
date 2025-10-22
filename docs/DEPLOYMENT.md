# 🚀 Deployment Guide

This guide covers different deployment options for the Car Price Prediction project.

## 📋 Prerequisites

- Python 3.8 or higher
- All dependencies installed (`pip install -r requirements.txt`)
- Trained model artifacts in the `artifacts/` directory

## 🌐 Streamlit Cloud Deployment

### 1. Prepare Repository

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial project setup"
   git push origin main
   ```

2. **Create requirements.txt** (already created)
   - Ensure all dependencies are listed
   - Pin versions for reproducibility

3. **Create .streamlit/config.toml** (optional)
   ```toml
   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   ```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `src/streamlit/app.py`
6. Click "Deploy"

### 3. Configure Environment Variables

In Streamlit Cloud settings, add any required environment variables:
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

## 🐳 Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run

```bash
# Build image
docker build -t car-price-prediction .

# Run container
docker run -p 8501:8501 car-price-prediction
```

### 3. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  car-price-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./artifacts:/app/artifacts
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Run with:
```bash
docker-compose up -d
```

## ☁️ Cloud Platform Deployment

### Heroku

1. **Create Procfile**
   ```
   web: streamlit run src/streamlit/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**
   ```
   python-3.9.16
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### AWS EC2

1. **Launch EC2 instance**
2. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Run application**
   ```bash
   streamlit run src/streamlit/app.py --server.port=8501 --server.address=0.0.0.0
   ```

4. **Configure security group** to allow port 8501

### Google Cloud Platform

1. **Create App Engine configuration** (`app.yaml`):
   ```yaml
   runtime: python39
   
   entrypoint: streamlit run src/streamlit/app.py --server.port=8080 --server.address=0.0.0.0
   
   env_variables:
     STREAMLIT_SERVER_PORT: 8080
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

## 🔧 Production Considerations

### Security

- Add authentication if needed
- Use environment variables for sensitive data
- Implement rate limiting
- Add input validation

### Performance

- Use caching for model loading
- Implement connection pooling
- Monitor resource usage
- Add logging

### Monitoring

- Set up health checks
- Monitor application logs
- Track prediction accuracy
- Alert on errors

## 📊 Model Updates

### Automated Retraining

1. **Set up CI/CD pipeline**
2. **Schedule model retraining**
3. **Automated testing**
4. **Deploy new models**

### Manual Updates

1. **Train new model**
2. **Update artifacts**
3. **Redeploy application**
4. **Verify functionality**

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**
   - Check artifacts directory
   - Verify file paths
   - Ensure model is trained

2. **Import errors**
   - Check Python path
   - Verify dependencies
   - Update requirements.txt

3. **Port conflicts**
   - Change port number
   - Check if port is in use
   - Update configuration

### Debug Mode

Run with debug information:
```bash
streamlit run src/streamlit/app.py --logger.level=debug
```

## 📈 Scaling

### Horizontal Scaling

- Use load balancer
- Deploy multiple instances
- Implement session management

### Vertical Scaling

- Increase memory
- Add more CPU cores
- Optimize model size

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      run: echo "Deployment triggered"
```

## 📞 Support

For deployment issues:
- Check Streamlit documentation
- Review error logs
- Test locally first
- Contact support team
