from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Configuración del servidor
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Configuración de Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "documents"
    qdrant_use_grpc: bool = False
    
    # Configuración de Langfuse
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    # Configuración del LLM
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Instancia global de configuración
settings = Settings()