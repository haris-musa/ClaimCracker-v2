# ClaimCracker v2 - Project Overview

## Project Description

ClaimCracker is a fake news detection system that combines machine learning with a modern web interface. Originally developed as a Final Year Project, version 2 aims to be a production-ready demo showcasing modern ML and web development practices.

## Core Components

1. **ML System**

   - Text classification model for fake news detection
   - Preprocessing pipeline for news articles
   - Training infrastructure with Colab support

2. **Web Interface**

   - FastAPI backend with async support
   - Modern API design with OpenAPI docs
   - Production-ready error handling

3. **Deployment**
   - Hosted on Render.com (free tier)
   - Optimized for cloud deployment
   - CI/CD pipeline integration

## Dataset

- Location: `./Dataset/`
- Files:
  - Dataset.csv (7.3MB) - Combined dataset
  - Dataset - Real.csv (2.7MB) - Real news articles
  - Dataset - Fake.csv (2.1MB) - Fake news articles

## Technology Stack

- Python 3.11
- FastAPI + Uvicorn
- TensorFlow/Transformers
- Pandas + Scikit-learn
- Modern async patterns

## Development Status

- ✓ Environment setup (Python 3.11, Miniconda)
- ✓ Dependencies installed
- ✓ Project structure defined
- ⚡ Next: ML Pipeline Development

## Key Goals

1. Production-quality code with modern practices
2. Resource-optimized for free tier deployment
3. Easy to understand and maintain
4. Well-documented for showcase purposes
