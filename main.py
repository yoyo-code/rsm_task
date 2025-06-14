from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QueryRequest, QueryResponse, IngestResponse, HealthResponse
from rag_service import RAGService
from config import settings
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

# Inicializar servicio RAG
rag_service = RAGService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check"""
    try:
        # Verificar conexión a Qdrant
        collections = rag_service.vector_store_manager.client.get_collections()
        collection_info = rag_service.vector_store_manager.get_collection_info()
        
        if collection_info:
            status_detail = f"Conectado a Qdrant. Colección '{collection_info['name']}' con {collection_info['vectors_count']} vectores"
        else:
            status_detail = f"Conectado a Qdrant con {len(collections.collections)} colecciones"
        
        return HealthResponse(
            status="OK", 
            message=f"Servicio funcionando correctamente. {status_detail}"
        )
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_think_python():
    """Endpoint para scrapear e indexar el sitio web completo de Think Python y PEP-8 usando fragmentación semántica"""
    try:
        logger.info("Iniciando proceso de ingesta del sitio web Think Python y PEP-8")
        
        result = await rag_service.ingest_think_python_website()
        logger.info("Proceso de ingesta completado exitosamente")
        
        return result
        
    except Exception as e:
        logger.error(f"Error en ingesta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar sitio web: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Endpoint para hacer consultas usando RAG agentic"""
    try:
        logger.info(f"Procesando consulta: {request.question}")
        result = await rag_service.query_document(request.question)
        logger.info("Consulta procesada exitosamente")
        return result
    except Exception as e:
        logger.error(f"Error en consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")

@app.get("/collections/info")
async def get_collection_info():
    """Endpoint para obtener información de las colecciones de Qdrant"""
    try:
        # Usar la función correcta del vector store manager
        collection_info = rag_service.vector_store_manager.get_collection_info()
        
        return {
            "collections": [collection_info] if collection_info else []
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo info de colecciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/pages/info")
async def get_pages_info():
    """Endpoint adicional para obtener información sobre las páginas indexadas"""
    try:
        collection_info = rag_service.vector_store_manager.get_collection_info()
        
        if collection_info['vectors_count'] > 0:
            # Hacer una búsqueda de prueba para obtener muestra de metadatos
            test_results = rag_service.vector_store_manager.test_similarity_search("Python", k=5)
            
            unique_pages = set()
            for doc in test_results:
                unique_pages.add(doc.metadata.get('page', 'unknown'))
            
            return {
                "total_vectors": collection_info['vectors_count'],
                "sample_pages": list(unique_pages),
                "base_url": "https://allendowney.github.io/ThinkPython/",
                "fragmentation_type": "semantic"
            }
        else:
            return {
                "total_vectors": 0,
                "sample_pages": [],
                "message": "No hay páginas indexadas"
            }
        
    except Exception as e:
        logger.error(f"Error obteniendo info de páginas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)