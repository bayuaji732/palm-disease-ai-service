from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Palm Oil Disease AI Service")
app.include_router(router)
