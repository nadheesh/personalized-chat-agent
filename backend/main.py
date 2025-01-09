import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

# FastAPI app initialization
app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    openai_api_version="2024-06-01",
    azure_endpoint=os.environ["CHOREO_AZUREOPENAI_SERVICEURL"],
    openai_api_key=os.environ["CHOREO_AZUREOPENAI_AZURE_OPENAI_API_KEY"])

# Load values from environment variables
host = os.getenv("CHOREO_MFST_DB_HOSTNAME")
username = os.getenv("CHOREO_MFST_DB_USERNAME")
password = os.getenv("CHOREO_MFST_DB_PASSWORD")
port = os.getenv("CHOREO_MFST_DB_PORT", "11867")
dbname = os.getenv("CHOREO_MFST_DB_DATABASENAME")
collection_name = os.getenv("DB_COLLECTION_NAME", "my_docs")

connection = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{dbname}"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    async_mode=True
)

# Initialize OpenAI language model
llm = AzureChatOpenAI(
    deployment_name="choreo-ai-chat-4o",
    openai_api_version="2024-02-01",
    azure_endpoint=os.environ["CHOREO_AZUREOPENAI_SERVICEURL"],
    openai_api_key=os.environ["CHOREO_AZUREOPENAI_AZURE_OPENAI_API_KEY"])


# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: Optional[str] = ""


class ConversationRequest(BaseModel):
    message: str
    chat_history: List[Message]


@app.post("/add")
async def add_knowledge(content: str):
    """
    Use to add new information to the ask questions

    Args:
        content (text): Adde knowledge to the system.

    Returns:
        dict: A message indicating success.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(content)

        # Add chunks to vector store
        await vector_store.aadd_texts(chunks)
        return {"message": "Updated the knowledge base successfully."}
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Instructions for the LLM to reformulate the latest user question using chat history context.
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

# Instructions for the LLM to generate an answer to the user question using the provided context.
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following retrieved context to answer the question. \
Only refer to the provided context when forming your answer. \
If there isn't enough information to answer the question, \
simply state that you don't have sufficient information. \
Provide concise answers, structured neatly using simple Markdown.

{context}"""


@app.post("/ask")
async def ask_question(request: ConversationRequest):
    """
       By looking the conversation request answers to the latest user question.

       Args:
           request (ConversationRequest): The conversation request details.

       Returns:
           dict: The AI-generated response.

       Raises:
           HTTPException: If there's an error processing the request.
       """
    try:
        message = request.message

        # Convert chat history to a list of tuples containing roles and message content.
        chat_history = [(msg.role, msg.content) for msg in request.chat_history]

        # Initialize a retriever from the vector store.
        # Filtered by user ID and limiting to 5 results.
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        # Create a prompt template for the history aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create a retriever that uses question and chat history to retrieve context
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Create a prompt template for the answer generation
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create a chain that processes retrieved documents to generate an answer.
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create a RAG chain that combines history-aware retrieval and answer generation.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Invoke the RAG chain with the user's input and chat history to get the response.
        response = await rag_chain.ainvoke({"input": message, "chat_history": chat_history})
        return {"response": response['answer']}
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
