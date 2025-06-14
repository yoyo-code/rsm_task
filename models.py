from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """Modelo para las consultas entrantes"""
    question: str = Field(..., description="Pregunta a realizar sobre el documento")

class Source(BaseModel):
    """Modelo para las fuentes de información"""
    page: int = Field(..., description="Número de página")
    text: str = Field(..., description="Texto del pasaje relevante")

class QueryResponse(BaseModel):
    """Modelo para las respuestas de consultas"""
    answer: str = Field(..., description="Respuesta generada")
    sources: List[Source] = Field(..., description="Fuentes utilizadas para generar la respuesta")

class HealthResponse(BaseModel):
    """Modelo para el health check"""
    status: str = Field(..., description="Estado del servicio")
    message: str = Field(..., description="Mensaje descriptivo")

class IngestResponse(BaseModel):
    """Modelo para la respuesta de ingesta"""
    status: str = Field(..., description="Estado del proceso de ingesta")
    message: str = Field(..., description="Mensaje descriptivo")
    document_info: Optional[dict] = Field(None, description="Información del documento procesado")