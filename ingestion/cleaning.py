from langchain_core.documents import Document
import logging
import hashlib
import re
from typing import List, Set

logger = logging.getLogger(__name__)

class ContentCleaner:
    """Manejador de limpieza y filtrado de contenido"""
    
    def __init__(self):
        # Patrones para detectar contenido de navegación
        self.navigation_patterns = [
            r"skip to main content",
            r"back to top",
            r"ctrl \+ k",
            r"\.ipynb \.pdf",
            r"think python \.ipynb \.pdf",
            r"^skip to",
            r"main content",
            r"navigation",
            r"menu",
            r"breadcrumb"
        ]
    
    def _is_navigation_content(self, text: str) -> bool:
        """Detectar si el texto es contenido de navegación"""
        text_lower = text.lower().strip()
        
        # Verificar patrones de navegación
        for pattern in self.navigation_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Verificar si es muy corto y no contiene información útil
        if len(text_lower) < 50 and not any(word in text_lower for word in ['python', 'function', 'variable', 'class', 'method']):
            return True
            
        return False
    
    def _is_quality_chunk(self, text: str) -> bool:
        """Validar que el chunk tenga contenido de calidad"""
        text_clean = text.strip()
        
        # Filtros de calidad
        if len(text_clean) < 30:  # Muy corto
            return False
            
        if self._is_navigation_content(text_clean):  # Es navegación
            return False
            
        # Verificar que tenga contenido sustancial
        words = text_clean.split()
        if len(words) < 5:  # Muy pocas palabras
            return False
            
        # Verificar que no sea solo puntuación o caracteres especiales
        alphanumeric_chars = sum(c.isalnum() for c in text_clean)
        if alphanumeric_chars < len(text_clean) * 0.7:  # Menos del 70% alfanumérico
            return False
            
        return True
    
    def _get_content_hash(self, text: str) -> str:
        """Generar hash del contenido para detectar duplicados"""
        # Normalizar texto: minúsculas, sin espacios extra, sin puntuación extra
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def filter_and_deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Filtrar y deduplicar documentos basado en contenido"""
        seen_hashes: Set[str] = set()
        filtered_documents = []
        
        for doc in documents:
            text = doc.page_content
            
            # Verificar calidad del chunk
            if not self._is_quality_chunk(text):
                continue
            
            # Verificar duplicados
            content_hash = self._get_content_hash(text)
            if content_hash in seen_hashes:
                continue
                
            seen_hashes.add(content_hash)
            filtered_documents.append(doc)
        
        logger.info(f"Filtrados: {len(documents)} -> {len(filtered_documents)} documentos (eliminados {len(documents) - len(filtered_documents)} duplicados/navegación)")
        return filtered_documents