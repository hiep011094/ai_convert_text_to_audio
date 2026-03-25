"""Run server without lifespan TTS init for testing non-TTS endpoints"""
import uvicorn
import sys
sys.path.insert(0, ".")

# Patch lifespan to skip TTS init
import server
from contextlib import asynccontextmanager

@asynccontextmanager
async def quick_lifespan(app):
    print("🚀 Quick-start mode (no TTS warmup)")
    yield
    print("👋 Server stopped")

server.app.router.lifespan_context = quick_lifespan

if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=12345, log_level="info")
