import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

from ml_platform.orchestrator.routes import orch, router


@asynccontextmanager
async def lifespan(app: FastAPI):

    worker_task = asyncio.create_task( orch.run() )
    container_task = asyncio.create_task(orch.container_handler.monitor()) 

    try:
        yield

    finally:
        # Consider setting event to canceled 
        worker_task.cancel()
        container_task.cancel()


app = FastAPI(lifespan=lifespan)
app.include_router(router)