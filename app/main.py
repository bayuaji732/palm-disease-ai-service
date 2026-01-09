from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# Create FastAPI application
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["Palm Oil Disease"])


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {
        "message": "Palm Oil Disease AI Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }