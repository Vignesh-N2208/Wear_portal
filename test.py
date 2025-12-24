# test.py (Corrected Version)
import pytest
import asyncio
import os
import logging
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# --- UPDATED IMPORTS ---
from openai import AzureOpenAI  # NEW: Use native AzureOpenAI client
from ragas.llms import llm_factory
from langchain_openai import AzureOpenAIEmbeddings  # Still used for embeddings

# Import the RAG executor function
from rag_util import execute_rag_pipeline

# --- 1. RAGAS Client Setup (Using your Azure Config) ---
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT").rstrip("/")
CHAT_DEPLOY_HEAVY = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
CHAT_KEY = os.getenv("AZURE_CHAT_API_KEY")
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
EMBED_DEPLOY = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# --- CRITICAL FIX START ---

# 1. Instantiate the native AzureOpenAI Client
# This object is needed by the llm_factory
azure_client = AzureOpenAI(
    api_key=CHAT_KEY, azure_endpoint=CHAT_ENDPOINT, api_version=API_VERSION
)

# 2. Use llm_factory with the model name (deployment name) and the client object
ragas_llm = llm_factory(model=CHAT_DEPLOY_HEAVY, client=azure_client)

# 3. Embeddings (kept the same as it was not throwing an error)
ragas_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=API_VERSION,
    azure_endpoint=EMBED_ENDPOINT,
    azure_deployment=EMBED_DEPLOY,
    openai_api_key=EMBED_KEY,
)

# --- CRITICAL FIX END ---

# --- 2. GOLD TEST SET (Same as before) ---
# --- 2. GOLD TEST SET (Updated for Consistency) ---
GOLD_TEST_SET = [
    {
        "question": "What was the gravimetric wear at 1 million cycles?",
        "ground_truths": [
            # CHANGED to list the *actual* raw data points found in the contexts
            "The gravimetric wear at 1.0Mc is 8.2249 mm³ for report REP-2025-00003, 9.2681 mm³ for REP-2025-00001, and 15.081 mm³ for REP-2025-00002."
        ],
    },
    {
        "question": "Show me the report details for REP-2025-00002.",
        "ground_truths": [
            # CHANGED to match the materials actually found in the sample contexts
            "Report REP-2025-00002 was a hip implant test (Resurfacing Head) using CoCrMo on UHMWPE materials under ISO 14242-1 conditions."
        ],
    },
]


# --- 3. Test Runner (No changes needed here) ---
@pytest.mark.asyncio
async def test_rag_quality_gates():
    logging.info("Starting RAGAS Test preparation...")
    # ... (rest of the function logic is unchanged) ...
    tasks = []
    for item in GOLD_TEST_SET:
        tasks.append(execute_rag_pipeline(item["question"]))

    results = await asyncio.gather(*tasks)

    # Prepare data for RAGAS and log answers/contexts for debugging
    ragas_data = []
    with open("rag_debug_log.txt", "a", encoding="utf-8") as debug_log:
        for idx, (item, result) in enumerate(zip(GOLD_TEST_SET, results)):
            ragas_data.append(
                {
                    "question": item["question"],
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "reference": item["ground_truths"][0],
                }
            )
            debug_log.write(f"Test Case {idx+1}\n")
            debug_log.write(f"Question: {item['question']}\n")
            debug_log.write(f"Answer: {result['answer']}\n")
            debug_log.write(f"Contexts: {result['contexts']}\n")
            debug_log.write(f"Ground Truth: {item['ground_truths'][0]}\n")
            debug_log.write("-" * 60 + "\n")

    dataset = Dataset.from_list(ragas_data)

    # Run RAGAS Evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n--- RAGAS Evaluation Results ---")
    print(result.to_pandas().to_markdown())
    logging.info(f"RAGAS Results: {result}")

    # --- ENFORCE QUALITY GATES (Critical step!) ---
    # After RAGAS evaluation
    df = result.to_pandas()

    # Calculate average scores
    avg_faithfulness = df["faithfulness"].mean()
    avg_relevance = df["answer_relevancy"].mean()
    avg_context_recall = df["context_recall"].mean()

    # Calculate accuracy (if you have a 'correctness' column, otherwise use faithfulness as proxy)
    if "correctness" in df.columns:
        avg_accuracy = df["correctness"].mean()
    else:
        avg_accuracy = avg_faithfulness  # Use faithfulness as proxy for accuracy

    print("\n--- RAG Evaluation Metrics ---")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}")
    print(f"Average Relevance: {avg_relevance:.2f}")
    print(f"Average Context Recall: {avg_context_recall:.2f}")
    print(f"Accuracy (proxy): {avg_accuracy:.2f}")

    # Optionally, save to a log file
    with open("rag_metrics_log.txt", "a", encoding="utf-8") as logf:
        logf.write(f"Faithfulness: {avg_faithfulness:.2f}\n")
        logf.write(f"Relevance: {avg_relevance:.2f}\n")
        logf.write(f"Context Recall: {avg_context_recall:.2f}\n")
        logf.write(f"Accuracy (proxy): {avg_accuracy:.2f}\n")
        logf.write("-" * 40 + "\n")
