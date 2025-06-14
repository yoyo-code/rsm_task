from models import QueryResponse, Source, IngestResponse
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
from agentic_rag import AgenticRAG
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """Servicio principal para manejar las operaciones RAG con logging detallado"""
    
    def __init__(self):
        logger.info("Inicializando RAGService...")
        
        self.vector_store_manager = VectorStoreManager()
        self.document_processor = DocumentProcessor()
        self.agentic_rag = AgenticRAG(self.vector_store_manager)
        
        logger.info("RAG Service inicializado correctamente")
    
    async def ingest_document(self, pdf_path: str) -> IngestResponse:
        """Procesar e indexar documento en Qdrant con logging detallado"""
        if not pdf_path:
            raise ValueError("Se requiere un archivo PDF para la ingesta")
        
        logger.info(f"=== INICIANDO INGESTA ===")
        logger.info(f"Archivo PDF: {pdf_path}")
        
        try:
            # Procesar PDF
            logger.info("Paso 1: Procesando PDF...")
            documents, doc_ids = self.document_processor.process_pdf(pdf_path)
            logger.info(f"PDF procesado: {len(documents)} documentos creados")
            
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
            test_results = self.vector_store_manager.test_similarity_search("test", k=1)
            logger.info(f"Prueba de búsqueda: {len(test_results)} resultados")
            
            document_info = {
                "source": pdf_path,
                "total_chunks": len(documents),
                "vector_store": "Qdrant",
                "collection": self.vector_store_manager.collection_name,
                "vectors_indexed": collection_info['vectors_count']
            }
            
            logger.info(f"=== INGESTA COMPLETADA EXITOSAMENTE ===")
            
            return IngestResponse(
                status="success",
                message="Documento procesado e indexado correctamente en Qdrant",
                document_info=document_info
            )
            
        except Exception as e:
            logger.error(f"=== ERROR EN INGESTA ===")
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
                    page=source_data.get("page", 1),
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