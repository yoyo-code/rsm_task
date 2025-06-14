from models.schemas import QueryResponse, Source, IngestResponse
from services.vector_store import VectorStoreManager
from ingestion.pipeline import IngestionPipeline
from agent.agentic_rag import AgenticRAG
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """Servicio principal para manejar las operaciones RAG con páginas web"""
    
    def __init__(self):
        logger.info("Inicializando RAGService para procesamiento web...")
        
        self.vector_store_manager = VectorStoreManager()
        self.ingestion_pipeline = IngestionPipeline()
        self.agentic_rag = AgenticRAG(self.vector_store_manager)
        
        logger.info("RAG Service inicializado correctamente para web scraping")
    
    async def ingest_think_python_website(self) -> IngestResponse:
        """Procesar e indexar todo el sitio web de Think Python en Qdrant"""
        logger.info(f"=== INICIANDO INGESTA WEB THINK PYTHON ===")
        
        try:
            # Procesar sitio web 
            logger.info("Paso 1: Scrapeando y procesando sitio web...")
            documents, doc_ids = await self.ingestion_pipeline.process_all_websites_async()
            logger.info(f"Sitio web procesado: {len(documents)} documentos creados")
            
            # Indexar en Qdrant
            logger.info("Paso 2: Indexando en Qdrant...")
            self.vector_store_manager.create_vectorstore_from_documents(documents, doc_ids)
            
            # Verificar indexación
            logger.info("Paso 3: Verificando indexación...")
            collection_info = self.vector_store_manager.get_collection_info()
            logger.info(f"Verificación: {collection_info['vectors_count']} vectores indexados")
            
            # Configurar herramienta de recuperación en el agente
            logger.info("Paso 4: Configurando Agentic RAG...")
            self.agentic_rag.setup_retriever_tool()
            
            # Prueba de búsqueda para verificar funcionamiento
            logger.info("Paso 5: Probando búsqueda...")
            test_results = self.vector_store_manager.test_similarity_search("Python variables", k=1)
            logger.info(f"Prueba de búsqueda: {len(test_results)} resultados")
            
            # Contar páginas únicas procesadas
            unique_pages = set()
            for doc in documents:
                unique_pages.add(doc.metadata.get('page', 'unknown'))
            
            document_info = {
                "source": "Think Python Website + PEP-8",
                "total_chunks": len(documents),
                "unique_pages": len(unique_pages),
                "vector_store": "Qdrant",
                "collection": self.vector_store_manager.collection_name,
                "vectors_indexed": collection_info['vectors_count'],
                "base_urls": ["https://allendowney.github.io/ThinkPython/", "https://peps.python.org/pep-0008/"],
                "fragmentation_type": "semantic"
            }
            
            logger.info(f"=== INGESTA WEB COMPLETADA EXITOSAMENTE ===")
            
            return IngestResponse(
                status="success",
                message=f"Sitio web Think Python y PEP-8 procesados e indexados correctamente. {len(unique_pages)} páginas, {len(documents)} chunks semánticos",
                document_info=document_info
            )
            
        except Exception as e:
            logger.error(f"=== ERROR EN INGESTA WEB ===")
            logger.error(f"Error: {str(e)}")
            raise
    
    async def query_document(self, question: str) -> QueryResponse:
        """Procesar consulta usando RAG agentic con logging detallado"""
        logger.info(f"=== INICIANDO CONSULTA ===")
        logger.info(f"Pregunta: {question}")
        
        try:
            # Verificar que hay documentos indexados
            collection_info = self.vector_store_manager.get_collection_info()
            logger.info(f"Vectores disponibles: {collection_info['vectors_count']}")
            
            if collection_info['vectors_count'] == 0:
                raise ValueError("No hay documentos indexados. Debe ejecutar /ingest primero.")
            
            # Procesar consulta con el agente
            logger.info("Procesando con Agentic RAG...")
            result = await self.agentic_rag.process_query(question)
            
            # Convertir fuentes al formato esperado
            sources = []
            for source_data in result.get("sources", []):
                sources.append(Source(
                    page=source_data.get("page", "unknown"),
                    text=source_data.get("text", "")
                ))
            
            logger.info(f"=== CONSULTA COMPLETADA ===")
            logger.info(f"Respuesta generada: {result['answer'][:100]}...")
            logger.info(f"Fuentes: {len(sources)}")
            
            return QueryResponse(
                answer=result["answer"],
                sources=sources
            )
                
        except Exception as e:
            logger.error(f"=== ERROR EN CONSULTA ===")
            logger.error(f"Error: {str(e)}")
            raise