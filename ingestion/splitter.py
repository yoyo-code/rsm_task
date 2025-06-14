from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_core.documents import Document
from bs4 import Tag, BeautifulSoup
import logging
from typing import List

logger = logging.getLogger(__name__)

class SemanticSplitter:
    """Manejador de fragmentación semántica para HTML"""
    
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
    
    def _extract_page_title(self, html_content: str) -> str:
        """Extraer título de la página HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            return title.get_text().strip() if title else "Unknown Title"
        except:
            return "Unknown Title"
    
    def split_html_content(self, html_content: str, page_url: str) -> List[Document]:
        """Dividir contenido HTML en chunks semánticos"""
        try:
            logger.info(f"Dividiendo contenido de: {page_url}")
            
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
            logger.error(f"Error dividiendo contenido de {page_url}: {str(e)}")
            return []