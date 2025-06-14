from langchain_openai import OpenAIEmbeddings
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Servicio para manejar embeddings"""
    
    def __init__(self):
        self.embeddings = None
        self._initialize()
    
    def _initialize(self):
        """Inicializar embeddings de OpenAI"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"EmbeddingService inicializado con modelo: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Error inicializando embeddings: {str(e)}")
            raise
    
    def get_embeddings(self):
        """Obtener instancia de embeddings"""
        if not self.embeddings:
            raise ValueError("Embeddings no están inicializados")
        return self.embeddings