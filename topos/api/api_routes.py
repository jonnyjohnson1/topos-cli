# api_routes.py

from fastapi import APIRouter, HTTPException
import requests

router = APIRouter()


@router.post("/list_models")
async def list_models():
    url = "http://localhost:11434/api/tags"
    try:
        result = requests.get(url)
        if result.status_code == 200:
            return {"result": result.json()}
        else:
            raise HTTPException(status_code=404, detail="Models not found")
    except requests.ConnectionError:
        raise HTTPException(status_code=500, detail="Server connection error")
