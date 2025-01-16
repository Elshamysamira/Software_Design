from ctransformers import AutoModelForCausalLM
import chainlit as cl
import os
from typing import List
import torch
from sentence_transformers import SentenceTransformer, util

# Load the LLM and retrieval model
llm = None
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight sentence transformer

# Index for storing document embeddings and paths
document_embeddings = []
document_paths = []


def index_documents(folder_path: str):
    """Indexes all documents in the given folder."""
    global document_embeddings, document_paths
    document_embeddings.clear()
    document_paths.clear()

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith(".txt"):
            continue  # Skip non-text files
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        embedding = retrieval_model.encode(content, convert_to_tensor=True)
        document_embeddings.append(embedding)
        document_paths.append(file_path)
    print(f"Indexed {len(document_paths)} documents from {folder_path}")


def retrieve_documents(query: str, top_k: int = 3) -> List[str]:
    """Retrieves the top-k most relevant documents for a query."""
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    # Stack document embeddings into a single tensor
    stacked_embeddings = torch.stack(document_embeddings)  # Shape: [num_docs, embedding_dim]

    # Calculate cosine similarity between query and all document embeddings
    scores = util.cos_sim(query_embedding, stacked_embeddings)  # Shape: [1, num_docs]
    scores = scores.squeeze(0)  # Remove the batch dimension

    # Adjust top_k to avoid requesting more results than available
    top_k = min(top_k, len(document_embeddings))  # Ensure top_k is within bounds

    top_results = torch.topk(scores, k=top_k)  # Get top-k scores and indices
    return [document_paths[idx] for idx in top_results.indices.tolist()]


def get_prompt_with_context(instruction: str, context: List[str], history: List[str] = None) -> str:
    """Generates a prompt with retrieved context."""
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history and len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)} Now answer the question "
    prompt += f"{instruction}\n\n### Relevant Context:\n"
    prompt += "\n".join(context) + "\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    # Retrieve documents based on the query
    retrieved_docs = retrieve_documents(message.content, top_k=3)
    context = []
    for doc_path in retrieved_docs:
        with open(doc_path, "r", encoding="utf-8") as f:
            context.append(f.read())

    # Generate a prompt with the retrieved context
    prompt = get_prompt_with_context(message.content, context, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    # Load the LLM
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
    )
    # Index documents
    index_documents("/workspaces/local-llm-crash-course/documents")
