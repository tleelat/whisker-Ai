from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import router, meta_router
from server.settings import app_settings
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="WhiskerVision API",
    description="WhiskerVision is a production-ready FastAPI service for image classification.",
    version="1.0.0",
    contact={
        "name": "WhiskerVision",
        "url": "https://whiskervision.ai"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    redoc_url="/redoc",
    docs_url="/docs"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(meta_router)
app.include_router(router)