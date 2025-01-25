from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="ClaimCracker",
    description="Fake News Detection API",
    version="2.0.0"
)

class NewsInput(BaseModel):
    title: str
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to ClaimCracker API"}

@app.post("/predict")
async def predict_news(news: NewsInput):
    """
    Predict if a news article is real or fake.
    To be implemented with ML model integration.
    """
    try:
        # Placeholder for model prediction
        return {"status": "success", "message": "Prediction endpoint ready for integration"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 