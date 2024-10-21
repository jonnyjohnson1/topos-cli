
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import signal

router = APIRouter()

@router.post("/shutdown")
def shutdown(request: Request):
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse(content={"message": "Server shutting down..."})

@router.get("/health")
async def health_check():
    try:
        # Perform any additional checks here if needed
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/test")
async def test():
    return "hello world"
