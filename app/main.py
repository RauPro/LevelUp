
from fastapi import FastAPI
import uvicorn
from app.api.routes import router as api_router


app = FastAPI(
    title="LevelUp API",
    description="API for generating unique technical interview problems",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict:
    return {
        "message": "Welcome to the LevelUp API",
        "description": "Generate unique technical interview problems using AI",
        "docs": "/docs",
    }


app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
