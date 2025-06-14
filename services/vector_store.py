from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from core.config import settings
from services.embeddings import EmbeddingService
from langchain_core.documents import Document
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Maneja las operaciones del vector store con Qdrant siguiendo exactamente la documentación"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embeddings = self.embedding_service.get_embeddings()
        self.client = None
        self.vector_store = None
        self.collection_name = settings.qdrant_collection_name
        self._initialize()
    
    def _initialize(self):
        """Inicializar cliente de Qdrant"""
        try:
            # Conectar a Qdrant según la documentación
            if settings.qdrant_use_grpc:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    grpc_port=settings.qdrant_grpc_port,
                    prefer_grpc=True
                )
            else:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port
                )

            # Crear colección si no existe
            self._ensure_collection_exists()
            logger.info("VectorStoreManager inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando vector store: {str(e)}")
            raise
    
    def _ensure_collection_exists(self):
        """Asegurar que la colección existe en Qdrant"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=3072,  # Tamaño de embedding para text-embedding-3-large
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Colección {self.collection_name} creada")
            else:
                logger.info(f"Colección {self.collection_name} ya existe")
                
        except Exception as e:
            logger.error(f"Error verificando/creando colección: {str(e)}")
            raise
    
    def create_vectorstore_from_documents(self, documents: List[Document], ids: Optional[List[str]] = None):
        """Crear vector store desde documentos usando el patrón exacto de la documentación"""
        try:
            # Limpiar colección existente
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("Colección existente eliminada")
            except:
                logger.info("No había colección previa que eliminar")
            
            self._ensure_collection_exists()
            
            # Usar el patrón exacto de la documentación de Qdrant
            self.vector_store = QdrantVectorStore.from_documents(
                documents,
                self.embeddings,
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=self.collection_name,
                prefer_grpc=settings.qdrant_use_grpc
            )
            
            # Verificar que los documentos se indexaron correctamente
            count = self.client.count(collection_name=self.collection_name)
            logger.info(f"Vector store creado con {count.count} documentos indexados")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creando vector store desde documentos: {str(e)}")
            raise
    
    def connect_to_existing_collection(self):
        """Conectar a una colección existente usando el patrón de la documentación"""
        try:
            # Usar el patrón from_existing_collection de la documentación
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=f"http://{settings.qdrant_host}:{settings.qdrant_port}"
            )
            
            # Verificar conexión
            count = self.client.count(collection_name=self.collection_name)
            logger.info(f"Conectado exitosamente a colección con {count.count} vectores")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error conectando a colección existente: {str(e)}")
            raise
    
    def get_retriever(self, **kwargs):
        """Obtener retriever configurado"""
        if not self.vector_store:
            raise ValueError("Vector store no está inicializado. Debe ingestar documentos primero.")
        retriever = self.vector_store.as_retriever(**kwargs)
        return retriever
    
    def get_collection_info(self):
        """Obtener información detallada de la colección"""
        try:
            # Obtener información básica de la colección
            collection_info = self.client.get_collection(self.collection_name)
            
            # Obtener el conteo real de puntos/vectores usando el método correcto
            points_count = self.client.count(collection_name=self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": points_count.count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de colección: {str(e)}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "status": "error"
            }
    
    def test_similarity_search(self, query: str, k: int = 3):
        """Método de prueba para verificar que la búsqueda funciona"""
        try:
            if not self.vector_store:
                logger.error("Vector store no inicializado")
                return []
            
            results = self.vector_store.similarity_search(query, k=k)
            
            for i, doc in enumerate(results):
                logger.info(f"Resultado {i+1}: página {doc.metadata.get('page', 'N/A')}, contenido: {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de prueba: {str(e)}")
            return []