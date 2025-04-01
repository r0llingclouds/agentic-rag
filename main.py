from fastapi import FastAPI
from routes import router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI(
    title="AI Investment Advisor",
    description="Multi-agent system for Investment Advise using CrewAI and WatsonX",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Retail AI Investment Advisor API Using CrewAI & WatsonX.ai. Use /docs for API documentation."}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)