from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal
from config import settings
import logging
import re

logger = logging.getLogger(__name__)

# Variables globales para los modelos
response_model = None
grader_model = None
retriever_tool = None

def initialize_models():
    """Inicializar modelos globales usando ChatOpenAI"""
    global response_model, grader_model
    response_model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0
    )
    grader_model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0
    )

def custom_retriever_function(vector_store_manager):
    """Función retriever personalizada que mantiene metadatos"""
    def _retrieve(query: str) -> str:
        """Recuperar documentos y formatear con metadatos"""
        try:
            # Obtener retriever base
            retriever = vector_store_manager.get_retriever(k=4)
            # Recuperar documentos con metadatos
            documents = retriever.invoke(query)
            # Formatear respuesta incluyendo metadatos
            formatted_response = ""
            for i, doc in enumerate(documents):
                page_info = doc.metadata.get('page', 'N/A')
                source_info = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', i)
                formatted_response += f"[DOCUMENTO_{i+1}|PAGE_{page_info}|SOURCE_{source_info}|CHUNK_{chunk_id}]\n"
                formatted_response += doc.page_content + "\n\n"
            return formatted_response.strip()
        except Exception as e:
            logger.error(f"Error en retriever personalizado: {str(e)}")
            return f"Error retrieving documents: {str(e)}"
    return _retrieve

def setup_retriever_tool(vector_store_manager):
    """Configurar herramienta de recuperación personalizada"""
    global retriever_tool
    try:
        # Crear función retriever personalizada
        custom_retriever = custom_retriever_function(vector_store_manager)
        # Crear tool con función personalizada
        from langchain_core.tools import tool
        @tool
        def retrieve_document_content(query: str) -> str:
            """Search and return information from the uploaded document. Use this tool to answer questions about specific content, names, data, or details mentioned in the document."""
            return custom_retriever(query)
        retriever_tool = retrieve_document_content
        return retriever_tool
    except Exception as e:
        logger.error(f"Error configurando retriever tool: {str(e)}")
        raise

# Prompts 
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# Funciones de nodos exactamente como la documentación
def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def build_graph():
    """Construir el grafo exactamente como la documentación"""
    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    graph = workflow.compile()
    return graph

async def process_query_with_graph(question: str, graph) -> dict:
    """Procesar una consulta usando el grafo compilado"""
    try:
        final_answer = None
        all_messages = []
        sources = []
        for chunk in graph.stream({
            "messages": [
                {
                    "role": "user", 
                    "content": question
                }
            ]
        }):
            for node, update in chunk.items():
                if "messages" in update:
                    all_messages.extend(update["messages"])
                    last_message = update["messages"][-1]
                    if node == "generate_answer":
                        final_answer = last_message.content
                    elif node == "generate_query_or_respond":
                        if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                            final_answer = last_message.content
                    # Capturar sources desde mensajes de herramientas
                    if hasattr(last_message, 'name') and last_message.name == "retrieve_document_content":
                        sources = extract_sources_from_formatted_content(last_message.content)
        return {
            "answer": final_answer or "No se pudo generar una respuesta.",
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error procesando consulta con grafo: {str(e)}")
        raise

def extract_sources_from_formatted_content(content: str) -> list:
    """Extraer sources del contenido formateado con metadatos"""
    sources = []
    try:
        # Buscar bloques de documentos con patrón [DOCUMENTO_X|PAGE_Y|SOURCE_Z|CHUNK_W]
        document_pattern = r'\[DOCUMENTO_(\d+)\|PAGE_([^|]+)\|SOURCE_([^|]+)\|CHUNK_([^]]+)\]\n(.*?)(?=\[DOCUMENTO_|\Z)'
        matches = re.findall(document_pattern, content, re.DOTALL)
        for match in matches[:3]:  # Limitar a 3 fuentes
            doc_num, page_info, source_info, chunk_info, doc_content = match
            clean_content = doc_content.strip()
            source = {
                "page": str(page_info),  
                "text": clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
            }
            sources.append(source)
        if not sources:
            sources = extract_sources_fallback(content)
        return sources
    except Exception as e:
        logger.error(f"Error extrayendo sources formateadas: {str(e)}")
        return extract_sources_fallback(content)

def extract_sources_fallback(content: str) -> list:
    """Método de fallback para extraer sources"""
    sources = []
    try:
        # Dividir por párrafos o líneas
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        for i, paragraph in enumerate(paragraphs[:3]):
            if len(paragraph) > 20: 
                sources.append({
                    "page": str(i + 1),  
                    "text": paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
                })
        if not sources and content.strip():
            sources.append({
                "page": "1",  
                "text": content[:200] + "..." if len(content) > 200 else content
            })
        return sources
    except Exception as e:
        logger.error(f"Error en extract_sources_fallback: {str(e)}")
        return []

class AgenticRAG:
    """Wrapper para el sistema Agentic RAG"""
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.graph = None
        self.document_indexed = False
        initialize_models()
        self._check_existing_documents()
    def _check_existing_documents(self):
        """Verificar si ya hay documentos indexados y configurar automáticamente"""
        try:
            collection_info = self.vector_store_manager.get_collection_info()
            if collection_info and collection_info['vectors_count'] > 0:
                # Conectar a la colección existente
                self.vector_store_manager.connect_to_existing_collection()
                # Configurar el sistema
                self.setup_retriever_tool()
            else:
                logger.info("No se detectaron documentos existentes")
        except Exception as e:
            logger.error(f"Error verificando documentos existentes: {str(e)}")
            logger.info("Continuando sin configuración automática")
    def setup_retriever_tool(self):
        """Configurar herramienta de recuperación y construir grafo"""
        try:
            setup_retriever_tool(self.vector_store_manager)
            self.graph = build_graph()
            self.document_indexed = True
        except Exception as e:
            logger.error(f"Error configurando AgenticRAG: {str(e)}")
            raise
    async def process_query(self, question: str) -> dict:
        """Procesar una consulta"""
        if not self.document_indexed:
            raise ValueError("No hay documentos indexados. Debe ejecutar /ingest primero.")
        
        return await process_query_with_graph(question, self.graph)