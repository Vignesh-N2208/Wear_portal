# rag_utils.py (New File)
import aiohttp
import json
import uuid
from typing import List, Dict
from fastapi import HTTPException
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import QueryType

# Import all constants and clients from your main file
from api import (
    EMBED_ENDPOINT,
    EMBED_KEY,
    EMBED_DEPLOY,
    CHAT_ENDPOINT,
    CHAT_KEY,
    CHAT_DEPLOY_HEAVY,
    API_VERSION,
    search_client,
    get_db_data_for_comparison,
    infer_intent,
    TEST_ID_PAT,
)

system_prompt = (
   "You are a highly specialized Medical Device Biomechanics Analyst. Your task is to provide accurate, concise answers strictly based on the provided context, which consists of simulation wear reports."
   " **STRICTLY** base your response only on the provided context."
   "- **NEVER** use external knowledge."
   "- If the question can be answered, cite all relevant values and report IDs from the context."
   "- If the question cannot be answered from the provided context, you MUST respond with the exact phrase: \"The required information is not available in the current data.\""
)


# --- Utility Functions (Pulled from your code) ---


async def aoai_embeddings(
    session: aiohttp.ClientSession, texts: List[str]
) -> List[List[float]]:
    url = f"{EMBED_ENDPOINT}/openai/deployments/{EMBED_DEPLOY}/embeddings?api-version={API_VERSION}"
    async with session.post(
        url,
        headers={"api-key": EMBED_KEY, "Content-Type": "application/json"},
        json={"input": texts},
    ) as resp:
        if resp.status >= 400:
            raise HTTPException(
                status_code=resp.status,
                detail=f"Embeddings failed: {await resp.text()}",
            )
        data = await resp.json()
        return [item["embedding"] for item in data["data"]]


async def aoai_embed_one(session: aiohttp.ClientSession, text: str) -> List[float]:
    return (await aoai_embeddings(session, [text]))[0]


async def retrieve_context(
    query: str, session: aiohttp.ClientSession, k: int = 8
) -> List[Dict]:
    qvec = await aoai_embed_one(session, query)
    vq = VectorizedQuery(vector=qvec, k_nearest_neighbors=k, fields="contentVector")
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        query_type=QueryType.SIMPLE,
        select=[
            "id",
            "content",
            "source",
            "report_id",
            "standard",
            "implant_type",
            "design_variant",
            "material_pair",
        ],
        top=k,
    )
    docs = []
    for r in results:
        docs.append(r)
    return docs


# --- NON-STREAMING LLM for Testing (Required by RAGAS) ---
async def aoai_chat_completion_non_stream(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_message: str,
    context: str,
) -> str:
    """
    Non-streaming synchronous call to Azure OpenAI for RAG generation. Used for RAGAS.
    """
    url = (
        f"{CHAT_ENDPOINT}/openai/deployments/{CHAT_DEPLOY_HEAVY}/chat/completions"
        f"?api-version={API_VERSION}"
    )
    full_user_message = (
        f"User Query: {user_message}\n\n"
        f"Context from Index:\n---\n{context}\n---"
        "Based on the CONTEXT provided above, answer the USER QUERY... [rest of your prompt]"
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,  # Key difference from the streaming function
    }

    async with session.post(
        url,
        headers={"api-key": CHAT_KEY, "Content-Type": "application/json"},
        json=payload,
    ) as resp:
        if resp.status >= 400:
            raise HTTPException(
                status_code=resp.status,
                detail=f"Chat API failed: {await resp.text()}",
            )
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


# --- Core RAG Execution Logic (Required for Testing) ---
async def execute_rag_pipeline(user_msg_raw: str) -> dict:
    """
    Executes the full RAG path (DB check -> Search -> LLM Gen) non-streaming.
    """
    user_msg = user_msg_raw.lower()
    search_query = (
        TEST_ID_PAT.search(user_msg).group(0)
        if TEST_ID_PAT.search(user_msg)
        else user_msg_raw
    )
    intent = infer_intent(user_msg_raw)
    db_records = None
    rag_docs = []
    rag_context = ""

    async with aiohttp.ClientSession() as temp_session:
        # 1. DB Check Logic
        if intent["is_statistical"] or intent["is_comparison"] or intent["wants_table"]:
            db_records = await get_db_data_for_comparison(user_msg_raw)

        # 2. Retrieval Logic (Fallback)
        if not db_records:
            rag_docs = await retrieve_context(search_query, temp_session, k=8)

        # 3. Context Building Logic
        if db_records:
            db_context_list = [
                f"File: {d.get('file')}; Report ID: {d.get('report_id')}; Implant: {d.get('implant_type')}; Total Wear: {d.get('total_wear_5mc')} mm³"
                for d in db_records
            ]
            rag_context = (
                f"Structured Data from Azure SQL ({len(db_records)} records found):\n---\n"
                + "\n".join(db_context_list)
            )
        elif rag_docs:
            context_chunks = [
                f"[Source: {d.get('source')}, Report ID: {d.get('report_id')}] Content: {d.get('content')}"
                for d in rag_docs
            ]
            rag_context = (
                f"Vector Search Context ({len(rag_docs)} chunks found):\n---\n"
                + "\n---\n".join(context_chunks)
            )

        # 4. LLM Generation
        final_answer = ""
        if rag_context:
            final_answer = await aoai_chat_completion_non_stream(
                temp_session, system_prompt, user_msg_raw, rag_context
            )

        return {
            "question": user_msg_raw,
            "answer": final_answer,
            # Context for RAGAS evaluation
            "contexts": (
                [d.get("content") for d in rag_docs] if rag_docs else db_context_list
            ),
            "retrieval_method": "DB" if db_records else "RAG",
        }
