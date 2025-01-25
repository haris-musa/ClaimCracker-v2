# ClaimCracker v2

A machine learning-powered fake news detection system with a web interface.

## Project Structure

```
ClaimCracker-v2/
├── Dataset/           # News dataset (real and fake news)
├── src/              # Source code for ML components
├── models/           # Saved ML models
├── web/             # FastAPI web application
├── tests/           # Unit tests
└── config/          # Configuration files
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:

```bash
# To be implemented
python src/train_model.py
```

2. Run the web application:

```bash
cd web
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

- GET `/`: Welcome message
- POST `/predict`: Predict if a news article is real or fake

## Development

- Use `black` for code formatting
- Run tests with `pytest`
- Follow PEP 8 style guidelines
