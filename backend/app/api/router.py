from fastapi import APIRouter

from app.api.routes import forecast, health, overpass, planning, tram_graph

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(forecast.router)
api_router.include_router(tram_graph.router)
api_router.include_router(overpass.router)

api_router.include_router(planning.router)
