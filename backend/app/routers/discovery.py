from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ai_discovery_service import ai_discovery_service

router = APIRouter()

class QueryIn(BaseModel):
    query: str

@router.post("/query")
async def discover(payload: QueryIn, top_n: int = 12):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Describe what you want to watch.")
    if len(payload.query) > 500:
        raise HTTPException(status_code=400, detail="Please keep the request under 500 characters.")
    return await ai_discovery_service.discover(payload.query.strip(), top_n=max(1, min(top_n, 30)))
