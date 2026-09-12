# llm_rag_service.py

import os
import shutil
import re
import hashlib
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# --- TOON Import ---
import toon  # Ensure you run: pip install python-toon  

# --- NATIVE OPENAI IMPORT ---
from openai import AsyncOpenAI

# --- LangChain Core (Retained strictly for Retrieval/Vector math) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Custom Retriever Imports ---
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Native OpenAI Async Client pointing to OpenRouter
# openai_client = AsyncOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY")
# )

# Initialize Native OpenAI Async Client pointing to OpenRouter
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"), # Make sure this is your real OpenRouter key!
    default_headers={
        "HTTP-Referer": "http://localhost:8000", # Helps OpenRouter identify the request
        "X-Title": "SQL Boomer: Making Database interactions easy",
    }
)

# ==============================================================================
# 1. CONFIGURATION & LOCAL FILE SYNC LOGIC (TOON EDITION)
# ==============================================================================
RAW_SCHEMA_FILENAME = "Table_Schema.toon"
GLOSSARY_FILENAME = "Business_Glossary.toon"

PERSISTENT_SCHEMA_DIR = "Schema_Store"
PERSISTENT_GLOSSARY_DIR = "Glossary_Store"

os.makedirs(PERSISTENT_SCHEMA_DIR, exist_ok=True)
os.makedirs(PERSISTENT_GLOSSARY_DIR, exist_ok=True)

PERSISTENT_SCHEMA_PATH = os.path.join(PERSISTENT_SCHEMA_DIR, RAW_SCHEMA_FILENAME)
PERSISTENT_INDEX_PATH = os.path.join(PERSISTENT_SCHEMA_DIR, "faiss_schema_index")

PERSISTENT_GLOSSARY_PATH = os.path.join(PERSISTENT_GLOSSARY_DIR, GLOSSARY_FILENAME)
PERSISTENT_GLOSSARY_INDEX_PATH = os.path.join(PERSISTENT_GLOSSARY_DIR, "faiss_glossary_index")

DEPLOYMENT_SCHEMA_PATH = os.path.join(".", RAW_SCHEMA_FILENAME)
DEPLOYMENT_GLOSSARY_PATH = os.path.join(".", GLOSSARY_FILENAME)

def calculate_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except FileNotFoundError:
        return None

def sync_config_files() -> bool:
    schema_updated = False
    if os.path.exists(DEPLOYMENT_SCHEMA_PATH):
        if calculate_file_hash(DEPLOYMENT_SCHEMA_PATH) != calculate_file_hash(PERSISTENT_SCHEMA_PATH):
            shutil.copy2(DEPLOYMENT_SCHEMA_PATH, PERSISTENT_SCHEMA_PATH)
            # Force Schema Rebuild immediately
            get_schema_retriever(force_rebuild=True) 

    if os.path.exists(DEPLOYMENT_GLOSSARY_PATH):
        if calculate_file_hash(DEPLOYMENT_GLOSSARY_PATH) != calculate_file_hash(PERSISTENT_GLOSSARY_PATH):
            shutil.copy2(DEPLOYMENT_GLOSSARY_PATH, PERSISTENT_GLOSSARY_PATH)
            # Force Glossary Rebuild immediately
            _build_glossary_retrievers(force_rebuild=True)
            
    return schema_updated

# ==============================================================================
# 2. LOCAL EMBEDDINGS & TRANSPARENT CUSTOM RETRIEVER
# ==============================================================================
print("Loading Local Embeddings...")
# embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-code") # better
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True}
)

def get_doc_id(doc: Document) -> str:
    unique_string = doc.page_content + str(doc.metadata.get("source", ""))
    return hashlib.sha256(unique_string.encode()).hexdigest()

class TransparentAsyncEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float] = Field(default_factory=list)
    c: int = 60
    top_k: int = 5 

    def _fuse_and_score(self, all_results: List[List[Document]]) -> List[Document]:
        weights = self.weights if self.weights else [1.0] * len(self.retrievers)
        fused_scores = {}
        doc_map = {}
        retriever_origins = {}

        for retriever_idx, (docs, weight) in enumerate(zip(all_results, weights)):
            for rank, doc in enumerate(docs):
                doc_id = get_doc_id(doc)
                if doc_id not in doc_map:
                    doc_copy = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                    doc_map[doc_id] = doc_copy
                    fused_scores[doc_id] = 0.0
                    retriever_origins[doc_id] = []
                    
                score_contribution = weight * (1 / (rank + self.c))
                fused_scores[doc_id] += score_contribution
                
                retriever_origins[doc_id].append({
                    "retriever_index": retriever_idx,
                    "original_rank": rank,
                    "score_contribution": score_contribution
                })

        for doc_id, doc in doc_map.items():
            doc.metadata["ensemble_rrf_score"] = fused_scores[doc_id]
            doc.metadata["ensemble_provenance"] = retriever_origins[doc_id]

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        final_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs][:self.top_k]
        
        logger.info(f"Ensemble retrieved and fused {len(final_docs)} unique documents.")
        return final_docs

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        all_results = [retriever.invoke(query) for retriever in self.retrievers]
        return self._fuse_and_score(all_results)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        tasks = [retriever.ainvoke(query) for retriever in self.retrievers]
        all_results = await asyncio.gather(*tasks)
        return self._fuse_and_score(all_results)


_glossary_ensemble = None

def _build_glossary_retrievers(force_rebuild=False):
    global _glossary_ensemble
    
    if _glossary_ensemble is not None and not force_rebuild:
        return

    if not os.path.exists(PERSISTENT_GLOSSARY_PATH): 
        return

    with open(PERSISTENT_GLOSSARY_PATH, 'r', encoding='utf-8') as f:
        # 1. Decode the TOON file into a dictionary
        parsed_toon = toon.decode(f.read())
        
    # 2. THE FIX: Extract the actual list of objects using the 'glossary' root key
    data = parsed_toon.get("glossary", [])

    # Now 'e' is guaranteed to be a dictionary, so .get() will work perfectly!
    docs = [Document(
        page_content=f"Term: {e.get('term', '')}\nDefinition: {e.get('definition', '')}\nLogic: {e.get('logic_hint', '')}", 
        metadata={"term": e.get("term", "")}
    ) for e in data]

    if not docs: return

    if force_rebuild or not os.path.exists(PERSISTENT_GLOSSARY_INDEX_PATH):
        faiss_vectorstore = FAISS.from_documents(docs, embeddings)
        faiss_vectorstore.save_local(PERSISTENT_GLOSSARY_INDEX_PATH)
    else:
        faiss_vectorstore = FAISS.load_local(PERSISTENT_GLOSSARY_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    _glossary_ensemble = TransparentAsyncEnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        top_k=3
    )

# ==============================================================================
# 3. SCHEMA LOADING (TOON NATIVE)
# ==============================================================================
_global_vectorstore = None

def get_schema_retriever(force_rebuild=False):
    global _global_vectorstore
    
    # If it's already in RAM and we aren't forcing a rebuild, just return it
    if _global_vectorstore is not None and not force_rebuild:
        return _global_vectorstore.as_retriever(search_kwargs={"k": 15})

    if force_rebuild or not os.path.exists(PERSISTENT_INDEX_PATH):
        print("Building fresh Schema FAISS Index...")
        with open(PERSISTENT_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            raw_toon_text = f.read()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1500)
        docs = text_splitter.create_documents([raw_toon_text])
        
        _global_vectorstore = FAISS.from_documents(docs, embeddings)
        _global_vectorstore.save_local(PERSISTENT_INDEX_PATH)
    else:
        print("Loading existing Schema FAISS Index from disk...")
        _global_vectorstore = FAISS.load_local(PERSISTENT_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
    return _global_vectorstore.as_retriever(search_kwargs={"k": 15})
# ==============================================================================
# 4. PARSING & HELPER LOGIC
# ==============================================================================
def parse_rag_response(rag_answer: str) -> tuple[str | None, str, str]:
    sql_pattern = r"```sql\s*(.*?)\s*```"
    match_sql = re.search(sql_pattern, rag_answer, re.DOTALL | re.IGNORECASE)
    raw_sql = match_sql.group(1).strip() if match_sql else ("NO_SQL" if "NO_SQL" in rag_answer else None)

    tech_pattern = r"(?i)(?:\d+\.\s*)?\**\s*Technical Reasoning\s*\**\s*:?\s*(.*?)(?=(?:\d+\.\s*)?\**\s*Layman Explanation)"
    match_tech = re.search(tech_pattern, rag_answer, re.DOTALL)
    tech_summary = match_tech.group(1).strip() if match_tech else "Technical details not provided."

    gen_pattern = r"(?i)(?:\d+\.\s*)?\**\s*Layman Explanation\s*\**\s*:?\s*(.*?)(?=(?:\d+\.\s*)?\**\s*SQL Code|```sql|$)"
    match_gen = re.search(gen_pattern, rag_answer, re.DOTALL)
    gen_summary = match_gen.group(1).strip() if match_gen else "Summary not provided."

    return raw_sql, tech_summary, gen_summary

def _convert_lc_history_to_openai(history: list[BaseMessage]) -> list[dict]:
    """Helper to translate LangChain message objects to OpenAI dicts."""
    mapped = []
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        mapped.append({"role": role, "content": msg.content})
    return mapped

# ==============================================================================
# 5. AGENTIC FIREWALL & SQL GENERATION (NATIVE OPENAI)
# ==============================================================================
async def execute_agentic_workflow(user_input: str, history: list[BaseMessage], client_time: Optional[str] = None):
    if not client_time:
        client_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 1. ALWAYS SYNC FIRST (This automatically rebuilds FAISS if needed)
    sync_config_files()
    
    # 2. LOAD RETRIEVERS INTO RAM (If not already loaded)
    global _glossary_ensemble
    if _glossary_ensemble is None: 
        _build_glossary_retrievers()

    # Model configuration for OpenRouter
    REFORMULATOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    SQL_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

    # --- GLOSSARY RETRIEVAL ---
    glossary_context_str = "No specific business terms found."
    if _glossary_ensemble:
        relevant_docs = await _glossary_ensemble.ainvoke(user_input)
        if relevant_docs:
            glossary_context_str = "\n---\n".join([d.page_content for d in relevant_docs])

    # --- PHASE 1: REFORMULATOR ---
    reformulator_sys_prompt = f"""
    You are a strict query reformulation assistant. Rewrite the user's latest question into a standalone, detailed plain English question based on the chat history.

    *** CURRENT USER TIME ***
    {client_time}

    *** DETECTED BUSINESS TERMINOLOGY ***
    {glossary_context_str}
    *************************************

    RULES:
    1. TRANSLATE JARGON: Replace user slang with the strict 'Logic' from the terminology above.
    2. TIME LOGIC: If the user says "yesterday", formulate it contextually.
    3. LOGIC PERSISTENCE: If the user refines a query (e.g., "remove the date"), extract original filters from history and output ONE merged question.
    4. NO SQL GUESSING: Write purely in English. Do not write SQL code.
    5. CHAT OVERRIDE: If the user just says "Hi" or "Thanks", return exactly their input.
    """

    # Build the messages array natively
    reformulator_messages = [{"role": "system", "content": reformulator_sys_prompt}]
    reformulator_messages.extend(_convert_lc_history_to_openai(history))
    reformulator_messages.append({"role": "user", "content": user_input})

    reformulate_response = await openai_client.chat.completions.create(
        model=REFORMULATOR_MODEL,
        messages=reformulator_messages,
        temperature=0
    )
    # Fall back to the original user input if the model returns None
    standalone_question = reformulate_response.choices[0].message.content or user_input

    # --- PHASE 2: SQL GENERATOR ---
    schema_retriever = get_schema_retriever()
    schema_docs = schema_retriever.invoke(standalone_question)
    schema_context = "\n\n".join(doc.page_content for doc in schema_docs)

    sql_sys_prompt = f"""
    You are an expert Text-to-SQL translator specialized strictly in **SQLite**.
    You will be provided with schema context in TOON (Token-Oriented Object Notation). Read it carefully.

    *** CONVERSATION BYPASS ***
    If the user input is just a greeting or general chat:
    1. Technical Reasoning: Conversational Interaction
    2. Layman Explanation: [Your polite response here]
    3. SQL Code: NO_SQL

    STRICT SQLITE RULES:
    1. NO HALLUCINATIONS: Use ONLY the columns defined in the TOON schema.
    2. ALIAS EVERYTHING: Use table aliases (e.g., `FROM orders AS o`).
    3. TIMEZONE/DATES: Use `date('now', 'localtime')` for today.
    4. FUZZY MATCHING: Use `LIKE '%term%'` for strings unless strict equality is demanded.
    5. SINGLE QUERY: Return exactly ONE query. Refuse mixed requests (lists + counts).
    6. FORMATTING (CRITICAL): You MUST place a space or newline before all major SQL keywords (WHERE, JOIN, GROUP BY, ORDER BY). Do NOT merge words (e.g., never output "LotIDfkWHERE").

    FORMAT:
    1. Technical Reasoning
    2. Layman Explanation
    3. ```sql ... ```

    SCHEMA CONTEXT:
    {schema_context}
    """

    sql_messages = [
        {"role": "system", "content": sql_sys_prompt},
        {"role": "user", "content": standalone_question}
    ]

    sql_response = await openai_client.chat.completions.create(
        model=SQL_MODEL,
        messages=sql_messages,
        temperature=0
    )
    # Fall back to a NO_SQL flag so the API handles it gracefully instead of crashing
    raw_response = sql_response.choices[0].message.content or "1. Technical Reasoning: API Failure\n2. Layman Explanation: The AI model failed to return a response.\n3. ```sql\nNO_SQL\n```"

    # Update state and attach the CLIENT'S exact timestamp
    human_msg = HumanMessage(content=user_input)
    human_msg.additional_kwargs["standalone_query"] = standalone_question
    human_msg.additional_kwargs["timestamp"] = client_time # <-- Locks in User's Time

    ai_msg = AIMessage(content=raw_response)
    ai_msg.additional_kwargs["timestamp"] = client_time    # <-- Locks in User's Time

    updated_history = history + [human_msg, ai_msg]

    return raw_response, updated_history

# ==============================================================================
# 6. SMART SUGGESTIONS (NATIVE OPENAI)
# ==============================================================================
async def generate_smart_suggestions(history_data: list[dict]) -> list[str]:
    user_queries = [
        item.get('standalone_query') for item in history_data[-50:] 
        if item.get('role') == 'user' and item.get('standalone_query')
    ]
    
    default_suggestions = ["Show me all orders.", "List the active clients.", "Count orders by status.", "Recent high-value sales."]
    if not user_queries: return default_suggestions
    
    prompt = f"""
    Analyze the user's historical queries: {json.dumps(user_queries)}
    Generate exactly 4 smart, distinct plain-English follow-up questions they might ask next.
    Return ONLY a raw JSON array of 4 strings. No markdown.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3
        )
        content = re.sub(r'^```json\s*|```\s*$', '', response.choices[0].message.content.strip(), flags=re.IGNORECASE)
        return json.loads(content)[:4]
    except Exception as e:
        logger.error(f"Suggestion Gen Error: {e}")
        return default_suggestions

# ==============================================================================
# 7. PREDICTIVE ACTION FOLLOW-UPS (NATIVE OPENAI)
# ==============================================================================
async def generate_predictive_followups(input_text: str, history: list[BaseMessage], session_id: str = "") -> list[str]:
    """
    Generates 4 clickable, forward-looking follow-up action buttons from the USER'S
    perspective based on the current question, trajectory, schema, and glossary.
    """
    default_actions = [
        "Group these results by community name.",
        "Filter for records created in the last 30 days.",
        "Show total base price across all builders.",
        "Sort these lots by highest base price."
    ]

    if not input_text or len(input_text.strip()) < 3:
        return default_actions

    try:
        # 1. Retrieve top schema context
        schema_retriever = get_schema_retriever()
        schema_docs = schema_retriever.invoke(input_text)[:4] if schema_retriever else []
        schema_summary = "\n---\n".join([d.page_content[:800] for d in schema_docs]) if schema_docs else "No schema found."

        # 2. Retrieve top glossary terminology
        global _glossary_ensemble
        if _glossary_ensemble is None:
            _build_glossary_retrievers()

        glossary_summary = "No business terminology found."
        if _glossary_ensemble:
            glossary_docs = await _glossary_ensemble.ainvoke(input_text)
            if glossary_docs:
                glossary_summary = "\n".join([f"- {d.page_content[:200]}" for d in glossary_docs[:4]])

        # 3. Extract user questions trajectory (last 5 user queries)
        user_queries = []
        for msg in history[-12:]:
            if isinstance(msg, HumanMessage):
                q = msg.additional_kwargs.get("standalone_query") or msg.content
                clean_q = re.sub(r"^\[Sent at:.*?\]\s*", "", q).strip()
                if clean_q:
                    user_queries.append(clean_q)

        user_trajectory = user_queries[-5:]
        trajectory_str = "\n".join([f"- {q}" for q in user_trajectory]) if user_trajectory else "None."

        # 4. Construct perspective-driven system prompt
        followup_sys_prompt = """You are an intelligent data copilot recommending the USER's next 4 actions.
Generate exactly 4 concise, actionable button labels written from the USER's viewpoint.

CRITICAL PERSPECTIVE RULES:
1. WRITE AS THE USER: These are buttons the user will click to send as their next message.
2. BAN ASSISTANT PHRASES: NEVER use "Would you like...", "Do you want me to...", "I can show you...", or question the user.
3. USE IMPERATIVE COMMANDS OR DIRECT INQUIRIES: E.g., "Group these results by...", "Filter to show only...", "What is the average base price for...", "Show lots on schedule hold".
4. SPECIFIC & GROUNDED: Use actual fields and concepts from the schema (Lots, Communities, Builders, Sales, Milestones, BasePrice, Region).
5. FORMAT: Return ONLY a valid JSON array of 4 strings. No markdown backticks, no markdown code blocks, no preamble."""

        followup_user_prompt = f"""Current Query:
{input_text}

User Query History Trajectory:
{trajectory_str}

Relevant Schema Context:
{schema_summary}

Relevant Business Glossary Terms:
{glossary_summary}

Generate 4 user-perspective follow-up action buttons as a JSON array of strings."""

        response = await openai_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {"role": "system", "content": followup_sys_prompt},
                {"role": "user", "content": followup_user_prompt}
            ],
            temperature=0.3
        )

        raw_content = response.choices[0].message.content or ""
        clean_content = re.sub(r'^```(?:json)?\s*', '', raw_content.strip(), flags=re.IGNORECASE)
        clean_content = re.sub(r'```\s*$', '', clean_content)
        
        parsed = json.loads(clean_content)
        if isinstance(parsed, list) and len(parsed) >= 4:
            return [str(item).strip() for item in parsed[:4]]

        return default_actions

    except Exception as exc:
        logger.warning(f"Predictive follow-ups generation fallback: {exc}")
        return default_actions