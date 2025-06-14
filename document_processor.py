import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
import logging
import os
from uuid import uuid4

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesa documentos PDF para el sistema RAG"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Asegurar que el directorio de uploads existe
        os.makedirs(settings.upload_dir, exist_ok=True)
    
    def extract_pages_from_pdf(self, pdf_path: str) -> list[dict]:
        """Extraer texto página por página manteniendo la información de página"""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
            
            doc.close()
            
            total_chars = sum(len(page["text"]) for page in pages)
            logger.info(f"Texto extraído del PDF: {total_chars} caracteres, {total_pages} páginas")
            
            return pages
                
        except Exception as e:
            logger.error(f"Error extrayendo páginas del PDF: {str(e)}")
            raise
    
    def create_documents_from_pages(self, pages: list[dict], source: str = "document.pdf") -> list[Document]:
        """Crear documentos de LangChain desde páginas, manteniendo la información correcta de página"""
        try:
            documents = []
            chunk_id = 0
            
            for page_data in pages:
                page_number = page_data["page_number"]
                page_text = page_data["text"]
                
                # Dividir el texto de esta página específica en chunks
                page_chunks = self.text_splitter.split_text(page_text)
                
                # Crear un documento para cada chunk de esta página
                for chunk in page_chunks:
                    if chunk.strip():  # Solo crear documento si el chunk tiene contenido
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": source,
                                "chunk_id": chunk_id,
                                "page": page_number
                            }
                        )
                        documents.append(doc)
                        chunk_id += 1
            
            # Agregar total_chunks a todos los documentos
            for doc in documents:
                doc.metadata["total_chunks"] = len(documents)
            
            logger.info(f"Creados {len(documents)} documentos desde {len(pages)} páginas")
            
            # Log de verificación de chunks por página
            page_counts = {}
            for doc in documents:
                page = doc.metadata["page"]
                page_counts[page] = page_counts.get(page, 0) + 1
            
            logger.info(f"Chunks por página: {page_counts}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error creando documentos desde páginas: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: str) -> tuple[list[Document], list[str]]:
        """Procesar un archivo PDF completo"""
        try:
            # Extraer páginas manteniendo la información de página
            pages = self.extract_pages_from_pdf(pdf_path)
            
            # Crear documentos procesando página por página
            documents = self.create_documents_from_pages(pages, source=os.path.basename(pdf_path))
            
            # Generar IDs únicos para cada documento
            doc_ids = [str(uuid4()) for _ in documents]
            
            return documents, doc_ids
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {str(e)}")
            raise