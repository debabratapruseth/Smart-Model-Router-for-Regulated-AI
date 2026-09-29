"""Start with: uvicorn app.main:app --reload (local demonstration only)."""
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import api
from app.core.config_loader import ROOT
from app.core.router import ModelRouter

load_dotenv(ROOT / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app):
    app.state.router = ModelRouter()
    yield


app = FastAPI(title="SMART MODEL ROUTER FOR BANKING", version="1.0.0", lifespan=lifespan)
app.include_router(api)
