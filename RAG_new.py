from ctransformers import AutoModelForCausalLM
import chainlit as cl
import os
from typing import List
import torch
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Initialize components
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence transformer
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)  # Chunker
vector_dimension = 384  # Embedding size for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(vector_dimension)  # FAISS Index for dense vector search
document_metadata = {}  # Map document ID to metadata

# LLM initialization placeholder
llm = None

# Functions


def add_to_faiss(embeddings: List[np.ndarray], metadata: List[dict]):
    """Add embeddings and their metadata to the FAISS index."""
    global index, document_metadata
    if not index.is_trained:
        index.train(np.array(embeddings))
    index.add(np.array(embeddings))
    for idx, meta in enumerate(metadata):
        document_metadata[index.ntotal - len(metadata) + idx] = meta


def index_documents(folder_path: str):
    """Indexes all documents in the given folder into FAISS."""
    global retrieval_model, index, document_metadata
    embeddings = []
    metadata = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith(".txt"):
            continue  # Skip non-text files
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split document into chunks
        chunks = text_splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = retrieval_model.encode(chunk, convert_to_tensor=False)
            embeddings.append(embedding)
            metadata.append({"file_path": file_path, "chunk_id": idx, "text": chunk})

    add_to_faiss(embeddings, metadata)
    print(f"Indexed {len(metadata)} chunks from {folder_path}")


def retrieve_documents(query: str, top_k: int = 3) -> List[dict]:
    """Retrieves the top-k most relevant chunks for a query."""
    global index, retrieval_model, document_metadata
    query_embedding = retrieval_model.encode(query, convert_to_tensor=False).reshape(1, -1)

    # Retrieve top-k results from FAISS
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Valid index
            meta = document_metadata[idx]
            results.append({**meta, "distance": dist})
    return results


def get_prompt_with_context(instruction: str, context: List[str], history: List[str] = None) -> str:
    """Generates a prompt with retrieved context."""
    system = "You are an AI assistant that gives helpful answers. Answer concisely and clearly."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history and len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)} Now answer the question "
    prompt += f"{instruction}\n\n### Relevant Context:\n"
    prompt += "\n".join(context) + "\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    global llm
    message_history = cl.user_session.get("message_history", [])
    msg = cl.Message(content="")
    await msg.send()

    # Retrieve documents based on the query
    retrieved_chunks = retrieve_documents(message.content, top_k=3)
    context = [chunk["text"] for chunk in retrieved_chunks]

    # Generate prompt with context
    prompt = get_prompt_with_context(message.content, context, message_history)

    # Generate response using LLM
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word

    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    global llm
    cl.user_session.set("message_history", [])

    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
    )

    # Index documents
    index_documents("/workspaces/local-llm-crash-course/documents")
