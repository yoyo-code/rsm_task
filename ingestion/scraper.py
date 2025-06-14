import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class WebScraper:
    """Scraper asíncrono para páginas web de Think Python y PEP-8"""
    
    def __init__(self):
        self.base_url = "https://allendowney.github.io/ThinkPython/"
        self.pep8_url = "https://peps.python.org/pep-0008/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_chapter_urls(self) -> List[str]:
        """Generar URLs de capítulos desde chap00.html hasta chap19.html"""
        urls = []
        for i in range(20):  # 0 a 19
            chapter_url = f"{self.base_url}chap{i:02d}.html"
            urls.append(chapter_url)
        
        logger.info(f"Generadas {len(urls)} URLs de capítulos")
        return urls
    
    async def scrape_page(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        """Scraper una página individual de forma asíncrona"""
        try:
            logger.info(f"Scrapeando: {url}")
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                html_content = await response.text()
                
                # También extraer título para logging
                soup = BeautifulSoup(html_content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else url
                
                logger.info(f"Scrapeado exitoso: {title_text}")
                
                return html_content, url
        
        except Exception as e:
            logger.error(f"Error scrapeando {url}: {str(e)}")
            raise
    
    async def scrape_pages_async(self, urls: List[str], max_concurrent: int = 10) -> List[Tuple[str, str]]:
        """Scrapear múltiples páginas de forma asíncrona"""
        scraped_pages = []
        
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        ) as session:
            # Crear semáforo para limitar concurrencia
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def scrape_with_semaphore(url):
                async with semaphore:
                    try:
                        return await self.scrape_page(session, url)
                    except Exception as e:
                        logger.warning(f"Saltando {url} debido a error: {str(e)}")
                        return None
            
            # Ejecutar todas las tareas de scraping
            tasks = [scrape_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrar resultados exitosos
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    scraped_pages.append(result)
        
        return scraped_pages
    
    def extract_pep8_sections(self, html_content: str, base_url: str) -> List[Tuple[str, str]]:
        """Extraer secciones de PEP-8 como páginas separadas"""
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = []
        
        # Buscar elementos section con ID
        section_elements = soup.find_all('section', id=True)
        
        for section in section_elements:
            section_id = section.get('id')
            if not section_id:
                continue
                
            # Crear URL con fragment
            section_url = f"{base_url}#{section_id}"
            
            # Obtener HTML de la sección completa
            section_html = str(section)
            
            sections.append((section_html, section_url))
            logger.info(f"Extraída sección: {section_url}")
        
        # Si no hay sections, buscar headers con ID
        if not sections:
            header_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], id=True)
            
            for header in header_elements:
                header_id = header.get('id')
                if not header_id:
                    continue
                    
                # Crear URL con fragment
                section_url = f"{base_url}#{header_id}"
                
                # Extraer contenido desde el header hasta el siguiente header del mismo nivel o superior
                section_html = self._extract_section_content_from_header(header, soup)
                
                sections.append((section_html, section_url))
                logger.info(f"Extraída sección desde header: {section_url}")
        
        return sections
    
    def _extract_section_content_from_header(self, header_element, soup) -> str:
        """Extraer contenido de una sección basada en un elemento header"""
        content_parts = [str(header_element)]
        
        # Determinar el nivel del header actual
        header_level = int(header_element.name[1]) if header_element.name.startswith('h') else 1
        
        # Buscar elementos siguientes hasta encontrar otro header del mismo nivel o superior
        current = header_element.next_sibling
        
        while current:
            if hasattr(current, 'name') and current.name:
                # Si es otro header del mismo nivel o superior, parar
                if (current.name.startswith('h') and 
                    len(current.name) == 2 and current.name[1].isdigit() and
                    int(current.name[1]) <= header_level):
                    break
                
                # Si es una sección, parar también
                if current.name == 'section':
                    break
                    
                content_parts.append(str(current))
            elif current.string:
                # Es texto
                content_parts.append(str(current))
            
            current = current.next_sibling
        
        return ''.join(content_parts)
    
    async def scrape_pep8(self) -> List[Tuple[str, str]]:
        """Scrapear PEP-8 y extraer secciones de forma asíncrona"""
        try:
            logger.info(f"Scrapeando PEP-8: {self.pep8_url}")
            
            # Scrapear la página completa
            pages = await self.scrape_pages_async([self.pep8_url], max_concurrent=1)
            
            if not pages:
                raise ValueError("No se pudo scrapear PEP-8")
            
            html_content, _ = pages[0]
            
            # Extraer secciones como páginas separadas
            sections = self.extract_pep8_sections(html_content, self.pep8_url)
            
            logger.info(f"PEP-8 scrapeado: {len(sections)} secciones extraídas")
            return sections
            
        except Exception as e:
            logger.error(f"Error scrapeando PEP-8: {str(e)}")
            return []
    
    async def scrape_all_chapters(self) -> List[Tuple[str, str]]:
        """Scrapear todos los capítulos de Think Python de forma asíncrona"""
        urls = self.get_chapter_urls()
        scraped_pages = await self.scrape_pages_async(urls, max_concurrent=10)
        
        logger.info(f"Think Python scrapeado: {len(scraped_pages)} páginas")
        return scraped_pages
    
    async def scrape_all_content(self) -> List[Tuple[str, str]]:
        """Scrapear todo el contenido: Think Python + PEP-8 de forma asíncrona"""
        all_pages = []
        
        # Scrapear Think Python y PEP-8 en paralelo
        logger.info("Iniciando scraping asíncrono de Think Python y PEP-8...")
        
        think_python_task = self.scrape_all_chapters()
        pep8_task = self.scrape_pep8()
        
        think_python_pages, pep8_sections = await asyncio.gather(
            think_python_task, 
            pep8_task
        )
        
        all_pages.extend(think_python_pages)
        all_pages.extend(pep8_sections)
        
        logger.info(f"Scrapeado completado: {len(all_pages)} páginas totales")
        logger.info(f"  - Think Python: {len(think_python_pages)} páginas")
        logger.info(f"  - PEP-8: {len(pep8_sections)} secciones")
        
        return all_pages