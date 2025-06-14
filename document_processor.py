from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from config import settings
from web_scraper import WebScraper
import logging
import os
from uuid import uuid4
from bs4 import Tag
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesa páginas web HTML para el sistema RAG usando HTMLSemanticPreservingSplitter SIMPLIFICADO"""
    
    def __init__(self):
        # Configurar headers para fragmentación semántica
        self.headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
            ("h5", "Header 5"),
            ("h6", "Header 6"),
        ]
        
        # HTMLSemanticPreservingSplitter para fragmentación semántica sin chunk_size fijo
        self.splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=self.headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            # Sin max_chunk_size para permitir fragmentación puramente semántica
            preserve_images=False,  # False porque usamos custom handler
            preserve_videos=True,
            elements_to_preserve=["table", "ul", "ol", "code", "pre", "blockquote"],
            denylist_tags=["script", "style", "head", "nav", "footer", "aside"],
            custom_handlers={"img": self._simple_image_handler},
        )
        
        self.web_scraper = WebScraper()
        
        # Asegurar que el directorio de uploads existe
        os.makedirs(settings.upload_dir, exist_ok=True)
    
    def _simple_image_handler(self, img_tag: Tag) -> str:
        """Handler simple para procesar imágenes sin LLM"""
        try:
            img_src = img_tag.get("src", "")
            img_alt = img_tag.get("alt", "No alt text provided")
            
            # Si la imagen no tiene src, retornar solo el alt text
            if not img_src:
                return f"[Image: {img_alt}]"
            
            # Convertir URL relativa a absoluta si es necesario
            if img_src.startswith('/'):
                img_src = f"https://allendowney.github.io{img_src}"
            elif not img_src.startswith('http'):
                img_src = f"https://allendowney.github.io/ThinkPython/{img_src}"
            
            # Retornar formato simple sin análisis LLM
            return f"[Image Alt Text: {img_alt} | Image Source: {img_src}]"
                
        except Exception as e:
            logger.error(f"Error en simple image handler: {str(e)}")
            return "[Image: Error processing image]"
    
    def _process_single_page(self, html_content_and_url: tuple) -> list[Document]:
        """Procesar una sola página HTML de forma simple"""
        html_content, page_url = html_content_and_url
        
        try:
            logger.info(f"Procesando página: {page_url}")
            
            # Dividir HTML usando HTMLSemanticPreservingSplitter
            page_documents = self.splitter.split_text(html_content)
            
            # Agregar metadata específico de la página
            for i, doc in enumerate(page_documents):
                # Mantener metadata existente y agregar info de página
                doc.metadata.update({
                    "source": page_url,
                    "page": page_url,  # Usar URL como "page" según solicitado
                    "page_title": self._extract_page_title(html_content)
                })
            
            logger.info(f"Creados {len(page_documents)} chunks semánticos para {page_url}")
            return page_documents
            
        except Exception as e:
            logger.error(f"Error procesando página {page_url}: {str(e)}")
            return []
    
    async def create_documents_from_html_pages_async(self, html_pages: list) -> list[Document]:
        """Crear documentos con paralelización simple usando ThreadPoolExecutor"""
        try:
            all_documents = []
            
            logger.info(f"Iniciando procesamiento paralelo simplificado de {len(html_pages)} páginas...")
            
            # Usar ThreadPoolExecutor para procesar páginas en paralelo
            with ThreadPoolExecutor(max_workers=6) as executor:
                # Enviar todas las tareas de procesamiento de páginas
                futures = [
                    executor.submit(self._process_single_page, page_data)
                    for page_data in html_pages
                ]
                
                # Recoger resultados conforme se completan
                for i, future in enumerate(futures):
                    try:
                        page_documents = future.result()
                        if page_documents:
                            all_documents.extend(page_documents)
                        logger.info(f"Completada página {i+1}/{len(html_pages)}")
                    except Exception as e:
                        logger.error(f"Error procesando página {i+1}: {str(e)}")
                        continue
            
            # Agregar chunk_id y total_chunks a todos los documentos
            for i, doc in enumerate(all_documents):
                doc.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(all_documents)
                })
            
            logger.info(f"Total documentos creados con paralelización simple: {len(all_documents)}")
            
            # Log de verificación de páginas
            page_counts = {}
            for doc in all_documents:
                page = doc.metadata["page"]
                page_counts[page] = page_counts.get(page, 0) + 1
            
            logger.info(f"Chunks por página: {len(page_counts)} páginas procesadas")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error creando documentos desde HTML: {str(e)}")
            raise
    
    def _extract_page_title(self, html_content: str) -> str:
        """Extraer título de la página HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            return title.get_text().strip() if title else "Unknown Title"
        except:
            return "Unknown Title"
    
    async def process_think_python_website_async(self) -> tuple[list[Document], list[str]]:
        """Procesar todo el sitio web de Think Python de forma asíncrona"""
        try:
            # Scrapear todas las páginas
            logger.info("Iniciando scraping asíncrono de Think Python...")
            html_pages = await self.web_scraper.scrape_all_chapters()
            
            if not html_pages:
                raise ValueError("No se pudieron scrapear páginas web")
            
            # Crear documentos con fragmentación semántica paralela simple
            documents = await self.create_documents_from_html_pages_async(html_pages)
            
            # Generar IDs únicos para cada documento
            doc_ids = [str(uuid4()) for _ in documents]
            
            return documents, doc_ids
            
        except Exception as e:
            logger.error(f"Error procesando sitio web de Think Python: {str(e)}")
            raise
        
    async def process_all_websites_async(self) -> tuple[list[Document], list[str]]:
        """Procesar Think Python y PEP-8 de forma simplificada"""
        try:
            # Scrapear todas las páginas (asíncrono)
            logger.info("Iniciando scraping asíncrono de Think Python y PEP-8...")
            html_pages = await self.web_scraper.scrape_all_content()
            
            if not html_pages:
                raise ValueError("No se pudieron scrapear páginas web")
            
            # Crear documentos con fragmentación semántica paralela simple
            documents = await self.create_documents_from_html_pages_async(html_pages)
            
            # Generar IDs únicos para cada documento
            doc_ids = [str(uuid4()) for _ in documents]
            
            return documents, doc_ids
            
        except Exception as e:
            logger.error(f"Error procesando sitios web: {str(e)}")
            raise
    
    # Mantener métodos síncronos para compatibilidad
    def create_documents_from_html_pages(self, html_pages: list) -> list[Document]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.create_documents_from_html_pages_async(html_pages))
    
    def process_think_python_website(self) -> tuple[list[Document], list[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_think_python_website_async())
        
    def process_all_websites(self) -> tuple[list[Document], list[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_all_websites_async())