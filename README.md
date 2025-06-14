# RAG Microservice con Think Python

Microservicio de Retrieval-Augmented Generation que procesa el sitio web Think Python con fragmentación semántica y patrón agentic usando LangGraph y Langfuse para trazabilidad.

## 🚀 Características

- **Web Scraping Asíncrono**: Scrapeado automático de Think Python y PEP-8
- **Fragmentación Semántica**: Usando HTMLSemanticPreservingSplitter de LangChain
- **RAG Agentic**: Sistema inteligente con LangGraph que incluye:
  - Evaluación de relevancia de documentos
  - Reescritura automática de consultas
  - Decisiones condicionales para búsqueda/respuesta
- **Vector Store**: Qdrant para almacenamiento vectorial optimizado
- **Observabilidad**: Trazabilidad completa con Langfuse
- **API REST**: Endpoints FastAPI para ingesta y consultas

## 📋 Requisitos Previos

- Docker y Docker Compose instalados
- Clave API de OpenAI
- Cuenta de Langfuse (opcional, para observabilidad)

## ⚡ Inicio Rápido

### 1. Clonar el repositorio

    git clone https://github.com/yoyo-code/rsm_task.git
    cd rsm_task

### 2. Configurar variables de entorno

    # Copiar el archivo de ejemplo
    cp .env.example .env

    # Editar el archivo .env con tus credenciales
    nano .env  # o tu editor preferido

**Variables requeridas**:

- `OPENAI_API_KEY`: Tu clave API de OpenAI
- `LANGFUSE_SECRET_KEY`: Tu clave secreta de Langfuse (opcional)
- `LANGFUSE_PUBLIC_KEY`: Tu clave pública de Langfuse (opcional)

### 3. Levantar los servicios

    # Construir y levantar todos los servicios
    docker-compose up --build

    # O en modo detached (background)
    docker-compose up --build -d

### 4. Verificar que los servicios estén funcionando

    # Health check
    curl http://localhost:8000/health

## 📖 Uso de la API

### Health Check

    curl -X GET "http://localhost:8000/health"

### Ingesta de Documentos

    curl -X POST "http://localhost:8000/ingest" \
      -H "Content-Type: application/json"

**Respuesta esperada**:

    {
      "status": "success",
      "message": "Sitio web Think Python y PEP-8 procesados e indexados correctamente. X páginas, Y chunks semánticos",
      "document_info": {
        "source": "Think Python Website + PEP-8",
        "total_chunks": 500,
        "unique_pages": 20,
        "vector_store": "Qdrant",
        "fragmentation_type": "semantic"
      }
    }

### Realizar Consultas

    curl -X POST "http://localhost:8000/query" \
      -H "Content-Type: application/json" \
      -d '{
        "question": "How do I reverse a list in Python?"
      }'

**Respuesta esperada**:

    {
      "answer": "Call `my_list.reverse()` — it reverses the list in place and returns `None`.",
      "sources": [
        {
          "page": "https://allendowney.github.io/ThinkPython/chap10.html",
          "text": "The reverse method reverses the elements of the list..."
        }
      ]
    }

## 🔧 Comandos Útiles

### Debugging

    # Entrar al contenedor de la API
    docker-compose exec api bash

    # Entrar al contenedor de Qdrant
    docker-compose exec qdrant bash

    # Ver el estado de los contenedores
    docker-compose ps

## 📊 Monitoreo con Langfuse

Si configuraste Langfuse, puedes monitorear:

1. **Trazas de consultas**: Ve el flujo completo desde la pregunta hasta la respuesta
2. **Métricas de retrieval**: Evalúa la calidad de la recuperación de documentos
3. **Costos de OpenAI**: Rastrea el uso de tokens y costos
4. **Latencia**: Mide los tiempos de respuesta

Accede a tu dashboard en: `https://us.cloud.langfuse.com`

## 🐛 Troubleshooting

### Error de conexión a Qdrant

    # Verificar que Qdrant esté corriendo
    curl http://localhost:6333/collections

    # Si no responde, reiniciar Qdrant
    docker-compose restart qdrant

### Error de API Key de OpenAI

    # Verificar que la variable de entorno esté configurada
    docker-compose exec api env | grep OPENAI_API_KEY

### Error de memoria en Qdrant

    # Verificar espacio en disco
    df -h

    # Ver logs de Qdrant
    docker-compose logs qdrant

### Limpiar y reiniciar completamente

    # Detener todo y eliminar volúmenes
    docker-compose down -v

    # Eliminar imágenes de Docker (opcional)
    docker-compose down --rmi all

    # Volver a construir
    docker-compose up --build

## 📁 Estructura del Proyecto

    rsm_task/
    ├── api/
    │   └── v1/
    │       └── router.py              # Endpoints de la API
    ├── agent/
    │   └── agentic_rag.py            # Sistema RAG con LangGraph
    ├── core/
    │   └── config.py                 # Configuración centralizada
    ├── ingestion/
    │   ├── scraper.py               # Web scraping asíncrono
    │   ├── splitter.py              # Fragmentación semántica
    │   ├── cleaning.py              # Limpieza de contenido
    │   └── pipeline.py              # Orquestador de ingesta
    ├── models/
    │   └── schemas.py               # Modelos Pydantic
    ├── services/
    │   ├── rag.py                   # Servicio principal RAG
    │   ├── vector_store.py          # Gestor de Qdrant
    │   └── embeddings.py            # Servicio de embeddings
    └── main.py                      # Aplicación FastAPI

## 🔗 Enlaces Útiles

- [Documentación de LangChain](https://python.langchain.com/docs/)
- [Documentación de Qdrant](https://qdrant.tech/documentation/)
- [Documentación de Langfuse](https://langfuse.com/docs)
- [Think Python Book](https://allendowney.github.io/ThinkPython/)
- [PEP-8 Style Guide](https://peps.python.org/pep-0008/)
