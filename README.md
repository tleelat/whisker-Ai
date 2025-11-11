# WhiskerVision API

WhiskerVision is a production-ready FastAPI service for image classification.

## Requirements
Python 3.9+

## Run locally
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Optional: copy `.env.example` to `.env` and customize (see Configuration)
3. Start the server:
   - `gunicorn -k uvicorn.workers.UvicornWorker -b <host>:<port> server.main:app`

## Run with Docker
Build:

`docker build -f Dockerfile.local -t <image_name> .`

Run:

`docker run -p <host_machine_port>:8080 <image_name>`

## API Docs
Swagger UI: `http://<host>:<port>/docs`  
ReDoc: `http://<host>:<port>/redoc`

## Endpoints
- `GET /health` — health check
- `GET /version` — service version
- `POST <API_PREFIX>/predict` — returns predicted class labels
- `POST <API_PREFIX>/predictproba` — returns class probabilities

Default `API_PREFIX`: `/api/v1/whiskervision`

## Configuration (environment variables)
All variables use the `MYAPP_` prefix to brand this service and avoid collisions.

- `MYAPP_ALLOWED_ORIGINS` (default: `https://whiskervision.ai,https://api.whiskervision.ai`) — CORS allowed origins (comma-separated)
- `MYAPP_API_PREFIX` (default: `/api/v1/whiskervision`) — API namespace
- `MYAPP_MODELS_DIR` (default: `models`) — directory containing `.pt` weights
- `MYAPP_MODEL_ARCH` (default: `resnet18`) — `resnet18` or `mobilenetv3`
- `MYAPP_REQUEST_MAX_ITEMS` (default: `10`) — max images per request

## Models
Place your trained weights under the models directory:
- `resnet18.pt`
- `mobilenetv3.pt`

If you rename files or directories, update `server/settings.py` and `server/routes.py` via env or defaults.

## Personalization & PII
- OpenAPI contact now uses brand and website only 
- `__author__` is set to `WhiskerVision`; 
- Tighten CORS further by setting `MYAPP_ALLOWED_ORIGINS` to your exact domains.

