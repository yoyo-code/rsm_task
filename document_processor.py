from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from config import settings
from web_scraper import WebScraper
import logging
from uuid import uuid4
from bs4 import Tag, BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesa páginas web HTML para el sistema RAG usando HTMLSemanticPreservingSplitter OPTIMIZADO"""
    
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
        
        # HTMLSemanticPreservingSplitter optimizado
        self.splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=self.headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            preserve_images=False,
            preserve_videos=True,
            elements_to_preserve=["table", "ul", "ol", "code", "pre", "blockquote", "p", "div"],
            # Expandir denylist para eliminar más elementos de navegación
            denylist_tags=["script", "style", "head", "nav", "footer", "aside", "header", "button", "form", "input"],
            custom_handlers={"img": self._simple_image_handler},
        )
        
        self.web_scraper = WebScraper()
        
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
    
    def _simple_image_handler(self, img_tag: Tag) -> str:
        """Handler simple para procesar imágenes sin LLM"""
        try:
            img_src = img_tag.get("src", "")
            img_alt = img_tag.get("alt", "No alt text provided")
            
            if not img_src:
                return f"[Image: {img_alt}]"
            
            if img_src.startswith('/'):
                img_src = f"https://allendowney.github.io{img_src}"
            elif not img_src.startswith('http'):
                img_src = f"https://allendowney.github.io/ThinkPython/{img_src}"
            
            return f"[Image Alt Text: {img_alt} | Image Source: {img_src}]"
                
        except Exception as e:
            logger.error(f"Error en simple image handler: {str(e)}")
            return "[Image: Error processing image]"
    
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
    
    def _extract_main_content(self, html_content: str) -> str:
        """Extraer solo el contenido principal de la página"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remover elementos de navegación y no deseados
            for tag in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', 'button', 'form']):
                tag.decompose()
            
            # Buscar contenido principal por selectores comunes
            main_selectors = [
                'main',
                '[role="main"]',
                '.content',
                '.main-content',
                '.article-content',
                'article',
                '.post-content'
            ]
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # Si no se encuentra contenido principal, usar body sin navegación
            if not main_content:
                main_content = soup.find('body')
            
            return str(main_content) if main_content else html_content
            
        except Exception as e:
            logger.error(f"Error extrayendo contenido principal: {str(e)}")
            return html_content
    
    def _filter_and_deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        """Filtrar y deduplicar documentos basado en contenido"""
        seen_hashes = set()
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
    
    def _process_single_page(self, html_content_and_url: tuple) -> list[Document]:
        """Procesar una sola página HTML de forma optimizada"""
        html_content, page_url = html_content_and_url
        
        try:
            logger.info(f"Procesando página: {page_url}")
            
            # Extraer solo contenido principal
            main_content = self._extract_main_content(html_content)
            
            # Dividir HTML usando HTMLSemanticPreservingSplitter
            page_documents = self.splitter.split_text(main_content)
            
            # Agregar metadata específico de la página
            for i, doc in enumerate(page_documents):
                doc.metadata.update({
                    "source": page_url,
                    "page": page_url,
                    "page_title": self._extract_page_title(html_content),
                    "chunk_index": i
                })
            
            logger.info(f"Creados {len(page_documents)} chunks para {page_url}")
            return page_documents
            
        except Exception as e:
            logger.error(f"Error procesando página {page_url}: {str(e)}")
            return []
    
    async def create_documents_from_html_pages_async(self, html_pages: list) -> list[Document]:
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
            filtered_documents = self._filter_and_deduplicate_documents(all_documents)
            
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
    
    def _extract_page_title(self, html_content: str) -> str:
        """Extraer título de la página HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            return title.get_text().strip() if title else "Unknown Title"
        except:
            return "Unknown Title"
    
    async def process_think_python_website_async(self) -> tuple[list[Document], list[str]]:
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
        
    async def process_all_websites_async(self) -> tuple[list[Document], list[str]]:
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
    def create_documents_from_html_pages(self, html_pages: list) -> list[Document]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.create_documents_from_html_pages_async(html_pages))
    
    def process_think_python_website(self) -> tuple[list[Document], list[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_think_python_website_async())
        
    def process_all_websites(self) -> tuple[list[Document], list[str]]:
        """Wrapper síncrono para compatibilidad"""
        return asyncio.run(self.process_all_websites_async())