from fastapi import FastAPI, APIRouter, Response, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from .core.config import settings
from .api import datasets, analysis, plots, quality
from .services import indexing, health
from .core.cache import analysis_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    print("Scanning filesystem and populating cache...")
    await indexing.populate_cache()
    print("Cache populated.")

    # Start background task for cache cleanup
    cleanup_task = asyncio.create_task(cache_cleanup_worker())

    yield
    # On shutdown
    cleanup_task.cancel()
    print("Application shutting down.")


async def cache_cleanup_worker():
    """Background worker to periodically clean up expired cache entries."""
    while True:
        await asyncio.sleep(300)  # Clean up every 5 minutes
        await analysis_cache.cleanup_expired()


app = FastAPI(
    title="Dataset Monitor API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(datasets.router, prefix="/datasets", tags=["Discovery"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(plots.router, prefix="/plots", tags=["Plots"])
api_router.include_router(quality.router, prefix="/quality", tags=["QualityCheck"])
# api_router.include_router(textblaster.router, prefix="/textblaster", tags=["TextBlaster"])


@app.get("/healthz", tags=["Health"])
def healthz():
    """Provides a simple health check of the API server."""
    return {"status": "ok", "message": "API is running"}


@app.get("/readyz", tags=["Health"])
async def readyz(response: Response):
    """Checks if the service is ready to accept traffic (dependencies are available)."""
    root_dir_ok, root_dir_msg = health.check_root_dir_exists()
    rabbitmq_ok, rabbitmq_msg = ("ok", "Not connected") # await health.check_rabbitmq_connection()

    checks = {
        "root_dir_check": {
            "status": "ok" if root_dir_ok else "error",
            "detail": root_dir_msg,
        },
        "rabbitmq_check": {
            "status": "ok" if rabbitmq_ok else "error",
            "detail": rabbitmq_msg,
        },
    }

    if root_dir_ok and rabbitmq_ok:
        return {"status": "ok", "checks": checks}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ok", "checks": checks}


app.include_router(api_router)
