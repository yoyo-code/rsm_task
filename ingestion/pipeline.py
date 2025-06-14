from langchain_core.documents import Document
from ingestion.scraper import WebScraper
from ingestion.splitter import SemanticSplitter
from ingestion.cleaning import ContentCleaner
import logging
from uuid import uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Orquestador del pipeline de ingesta - antes DocumentProcessor"""
    
    def __init__(self):
        self.web_scraper = WebScraper()
        self.splitter = SemanticSplitter()
        self.cleaner = ContentCleaner()
    
    def _process_single_page(self, html_content_and_url: Tuple[str, str]) -> List[Document]:
        """Procesar una sola página HTML de forma optimizada"""
        html_content, page_url = html_content_and_url
        
        try:
            logger.info(f"Procesando página: {page_url}")
            
            # Dividir HTML usando SemanticSplitter
            page_documents = self.splitter.split_html_content(html_content, page_url)
            
            logger.info(f"Creados {len(page_documents)} chunks para {page_url}")
            return page_documents
            
        except Exception as e:
            logger.error(f"Error procesando página {page_url}: {str(e)}")
            return []
    
    async def create_documents_from_html_pages_async(self, html_pages: List[Tuple[str, str]]) -> List[Document]:
        """Crear documentos con paralelización y filtrado optimizado"""
        try:
            all_documents = []
            
            logger.info(f"Iniciando procesamiento optimizado de {len(html_pages)} páginas...")
            
            # Usar ThreadPoolExecutor para procesar páginas en paralelo
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(self._process_single_page, page_data)
                    for page_data in html_pages
                ]
                
                for i, future in enumerate(futures):
                    try:
                        page_documents = future.result()
                        if page_documents:
                            all_documents.extend(page_documents)
                        logger.info(f"Completada página {i+1}/{len(html_pages)}")
                    except Exception as e:
                        logger.error(f"Error procesando página {i+1}: {str(e)}")
                        continue
            
            logger.info(f"Total documentos antes de filtrado: {len(all_documents)}")
            
            # Filtrar y deduplicar documentos
            filtered_documents = self.cleaner.filter_and_deduplicate_documents(all_documents)
            
            # Agregar chunk_id y total_chunks a documentos filtrados
            for i, doc in enumerate(filtered_documents):
                doc.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(filtered_documents)
                })
            
            logger.info(f"Total documentos después de filtrado: {len(filtered_documents)}")
            
            # Log de verificación de páginas
            page_counts = {}
            for doc in filtered_documents:
                page = doc.metadata["page"]
                page_counts[page] = page_counts.get(page, 0) + 1
            
            logger.info(f"Chunks de calidad por página: {len(page_counts)} páginas procesadas")
            for page, count in list(page_counts.items())[:5]:  # Mostrar primeras 5
                logger.info(f"  {page}: {count} chunks")
            
            return filtered_documents
            
        except Exception as e:
            logger.error(f"Error creando documentos desde HTML: {str(e)}")
            raise
    
    async def process_think_python_website_async(self) -> Tuple[List[Document], List[str]]:
        """Procesar todo el sitio web de Think Python de forma asíncrona"""
        try:
            logger.info("Iniciando scraping asíncrono de Think Python...")
            html_pages = await self.web_scraper.scrape_all_chapters()
            
            if not html_pages:
                raise ValueError("No se pudieron scrapear páginas web")
            
            documents = await self.create_documents_from_html_pages_async(html_pages)
            doc_ids = [str(uuid4()) for _ in documents]
            
            return documents, doc_ids
            
        except Exception as e:
            logger.error(f"Error procesando sitio web de Think Python: {str(e)}")
            raise
        
    async def process_all_websites_async(self) -> Tuple[List[Document], List[str]]:
        """Procesar Think Python y PEP-8 de forma optimizada"""
        try:
            logger.info("Iniciando scraping asíncrono de Think Python y PEP-8...")
            html_pages = await self.web_scraper.scrape_all_content()
            
            if not html_pages:
                raise ValueError("No se pudieron scrapear páginas web")
            
            documents = await self.create_documents_from_html_pages_async(html_pages)
            doc_ids = [str(uuid4()) for _ in documents]
            
            return documents, doc_ids
            
        except Exception as e:
            logger.error(f"Error procesando sitios web: {str(e)}")
            raise
    
    # Mantener métodos síncronos para compatibilidad
    def create_documents_from_html_pages(self, html_pages: List[Tuple[str, str]]) -> List[Document]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.create_documents_from_html_pages_async(html_pages))
    
    def process_think_python_website(self) -> Tuple[List[Document], List[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_think_python_website_async())
        
    def process_all_websites(self) -> Tuple[List[Document], List[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_all_websites_async())