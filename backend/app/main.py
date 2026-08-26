from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import movies, recommendations, sentiment, users, discovery
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Movie Recommendation Engine API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movies.router, prefix="/movies", tags=["movies"])
app.include_router(
    recommendations.router, prefix="/recommendations", tags=["recommendations"]
)
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(discovery.router, prefix="/discovery", tags=["discovery"])


@app.get("/health")
def health():
    return {"status": "ok"}
