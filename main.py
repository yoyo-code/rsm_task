from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import QueryRequest, QueryResponse
from core.config import settings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Microservice con Think Python",
    description="Microservicio de Retrieval-Augmented Generation que procesa el sitio web Think Python con fragmentación semántica y patrón agentic",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints de compatibilidad en la raíz (mantener endpoints originales)
@app.get("/health")
async def health_check_root():
    """Endpoint de health check en la raíz para compatibilidad"""
    from api.v1.router import health_check
    return await health_check()

@app.post("/ingest")
async def ingest_think_python_root():
    """Endpoint de ingesta en la raíz para compatibilidad"""
    from api.v1.router import ingest_think_python
    return await ingest_think_python()

@app.post("/query", response_model=QueryResponse)
async def query_document_root(request: QueryRequest):
    """Endpoint de consulta en la raíz para compatibilidad"""
    from api.v1.router import query_document
    return await query_document(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)