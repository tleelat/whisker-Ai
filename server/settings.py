from pydantic import BaseSettings
from enum import Enum

class ModelArch(str, Enum):
    MOBILENET_V3 = "mobilenetv3"
    RESNET18 = "resnet18"

class AppSettings(BaseSettings):
    model_arch: ModelArch = ModelArch.RESNET18
    request_max_items: int = 10
    allowed_origins: list[str] = ["https://whiskervision.ai", "https://api.whiskervision.ai"]
    api_prefix: str = "/api/v1/whiskervision"
    models_dir: str = "models"

    class Config:
        env_prefix = "MYAPP_"


app_settings = AppSettings()