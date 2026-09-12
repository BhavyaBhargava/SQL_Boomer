import json
import os
import re
import io
import csv
import asyncio
from datetime import datetime
from typing import Optional
from concurrent.futures import Executor
import pandas as pd
import pytz
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

HISTORY_DIR = "chat_history_logs"
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_server_time_str():
    """
    Returns current time as a string.
    Using Local for SQLite, but easily adaptable to specific zones.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_history_file_path(session_id: str) -> str:
    """Returns the local file path for a session's history."""
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def _to_serializable_history(messages: list[BaseMessage]) -> list[dict]:
    """
    Converts LangChain messages to a JSON-serializable list.
    - Preserves Timestamps.
    - Generates 'Turn-Based' chat_ids (User & AI share the same ID).
    - CLEANS the content (removes injected timestamps) to prevent duplication.
    """
    serializable = []
    current_time = get_server_time_str()

    # 1. Initialize counter based on existing IDs in the history
    current_turn_id = 0
    for msg in messages:
        existing_id = msg.additional_kwargs.get('chat_id')
        if existing_id and isinstance(existing_id, int):
            if existing_id > current_turn_id:
                current_turn_id = existing_id

    # 2. Process messages
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        
        timestamp = msg.additional_kwargs.get('timestamp', current_time)
        chat_id = msg.additional_kwargs.get('chat_id')

        # Logic to assign NEW turn IDs if missing
        if chat_id is None:
            if role == "user":
                current_turn_id += 1
                chat_id = current_turn_id
            else:
                chat_id = current_turn_id if current_turn_id > 0 else 1
            
            msg.additional_kwargs['chat_id'] = chat_id

        # --- FEATURE 6: Timestamp Deduplication ---
        # We strip the "[Sent at: ...]" prefix from the message before saving
        # so it doesn't infinitely loop into the text.
        raw_content = msg.content
        clean_content = re.sub(r"^\[Sent at:.*?\]\s*", "", raw_content)

        # --- FEATURE 7: Metadata Management ---
        # Catch the hidden standalone query to power the Smart Suggestions later
        standalone_query = None
        if role == "user":
            standalone_query = msg.additional_kwargs.get('standalone_query')

        payload = {
            "chat_id": chat_id,
            "role": role, 
            "content": clean_content, 
            "timestamp": timestamp 
        }
        
        if standalone_query:
            payload["standalone_query"] = standalone_query
            
        serializable.append(payload)

    return serializable

def load_history_from_local_file(session_id: str) -> list[BaseMessage]:
    """
    Loads history and INJECTS the timestamp into the visible content 
    so the LLM natively understands time relativity.
    """
    filepath = get_history_file_path(session_id)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                raw_history = json.load(f)
            except json.JSONDecodeError:
                return [] 

            # Sliding Window: Keep last 30 messages to prevent token overflow
            if len(raw_history) > 30: 
                raw_history = raw_history[-30:]

            messages = []
            for item in raw_history:
                ts = item.get("timestamp", "Unknown Time")
                c_id = item.get("chat_id")
                standalone_q = item.get("standalone_query")
                
                # --- FEATURE 6: Timestamp Injection ---
                # The LLM will literally read the time.
                content_with_time = f"[Sent at: {ts}] {item['content']}"

                if item["role"] == "user":
                    msg = HumanMessage(content=content_with_time)
                elif item["role"] == "assistant":
                    msg = AIMessage(content=content_with_time)
                
                # Store raw metadata so we don't lose it on re-save
                msg.additional_kwargs = {
                    "timestamp": ts,
                    "chat_id": c_id
                }

                # --- FEATURE 7: Re-inject standalone query metadata ---
                if standalone_q and item["role"] == "user":
                    msg.additional_kwargs["standalone_query"] = standalone_q
                
                messages.append(msg)
            
            return messages
    return []

def save_history_to_local_file(session_id: str, new_history: list[BaseMessage]):
    """Saves the updated LangChain message history to a local JSON file."""
    filepath = get_history_file_path(session_id)
    serializable_history = _to_serializable_history(new_history)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)

def get_raw_session_history(session_id: str) -> list[dict]:
    """
    Reads raw JSON history bypasses LangChain formatting.
    Used by the Smart Suggestions engine.
    """
    filepath = get_history_file_path(session_id)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return [] 
    return []

# ==============================================================================
# CONCURRENCY LOCKS & ASYNC WRAPPERS
# ==============================================================================
_session_locks: dict[str, asyncio.Lock] = {}

def get_session_lock(session_id: str) -> asyncio.Lock:
    """Returns or creates an asyncio.Lock specific to the given session_id."""
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]

def remove_cancelled_interaction(session_id: str) -> bool:
    """
    Removes the most recent interaction turn (highest chat_id) from the session log.
    Used when a user interrupts or cancels a query mid-stream to prevent incomplete state.
    """
    filepath = get_history_file_path(session_id)
    if not os.path.exists(filepath):
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if not data or not isinstance(data, list):
        return False

    turn_ids = [item.get("chat_id") for item in data if isinstance(item.get("chat_id"), int)]
    if not turn_ids:
        return False

    max_turn_id = max(turn_ids)
    pruned_data = [item for item in data if item.get("chat_id") != max_turn_id]

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(pruned_data, f, indent=2)
        return True
    except OSError:
        return False

async def load_history_async(session_id: str, executor: Optional[Executor] = None) -> list[BaseMessage]:
    """Thread-safe and session-locked async loading of session history."""
    async with get_session_lock(session_id):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, load_history_from_local_file, session_id)

async def save_history_async(session_id: str, new_history: list[BaseMessage], executor: Optional[Executor] = None) -> None:
    """Thread-safe and session-locked async saving of session history."""
    async with get_session_lock(session_id):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, save_history_to_local_file, session_id, new_history)

async def get_raw_session_history_async(session_id: str, executor: Optional[Executor] = None) -> list[dict]:
    """Thread-safe and session-locked async reading of raw JSON history."""
    async with get_session_lock(session_id):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, get_raw_session_history, session_id)

async def rollback_cancelled_turn_async(session_id: str, executor: Optional[Executor] = None) -> bool:
    """Thread-safe and session-locked async rollback of the last uncommitted turn."""
    async with get_session_lock(session_id):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, remove_cancelled_interaction, session_id)

# ==============================================================================
# BUSINESS GLOSSARY VERSIONING & PARSING
# ==============================================================================
def rotate_glossary_snapshots(persistent_dir: str):
    """
    Manages sliding-window backup versions for the Business Glossary:
    Moves active -> v1, v1 -> v2, and purges older v2.
    """
    v2_path = os.path.join(persistent_dir, "Business_Glossary_v2.json")
    v1_path = os.path.join(persistent_dir, "Business_Glossary_v1.json")
    active_path = os.path.join(persistent_dir, "Business_Glossary.json")

    if os.path.exists(v2_path):
        try:
            os.remove(v2_path)
        except OSError:
            pass

    if os.path.exists(v1_path):
        try:
            os.rename(v1_path, v2_path)
        except OSError:
            pass

    if os.path.exists(active_path):
        try:
            os.rename(active_path, v1_path)
        except OSError:
            pass

def parse_glossary_file(file_content: bytes, filename: str, end_row: Optional[int] = None) -> list[dict]:
    """
    Parses an uploaded Business Glossary file (.xlsx, .csv, or .json) into a standardized dictionary list.
    Automatically identifies the header row (scanning for 'term' / 'terms') to skip preamble text.
    Dynamically maps column variations and supports slicing with end_row.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.json':
        try:
            data = json.loads(file_content.decode('utf-8'))
            if isinstance(data, dict) and "glossary" in data:
                data = data["glossary"]
            if not isinstance(data, list):
                raise ValueError("JSON glossary must be an array or an object with a 'glossary' array.")
            return data
        except Exception as e:
            raise ValueError(f"Invalid JSON glossary file: {e}")

    # For CSV or Excel:
    if ext == '.csv':
        text_content = file_content.decode('utf-8', errors='replace')
        reader = csv.reader(io.StringIO(text_content))
        raw_rows = list(reader)
        if not raw_rows:
            return []
        max_len = max(len(r) for r in raw_rows)
        padded_rows = [r + [''] * (max_len - len(r)) for r in raw_rows]
        df = pd.DataFrame(padded_rows)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(io.BytesIO(file_content), header=None, dtype=str)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv, .xlsx, or .json.")

    if end_row is not None and end_row > 0:
        df = df.iloc[:end_row]

    # Find the header row containing 'term' or 'terms'
    header_idx = -1
    for idx, row in df.iterrows():
        row_values = [str(val).strip().lower() for val in row if pd.notna(val)]
        if any(v in ['term', 'terms', 'term name', 'business term'] for v in row_values):
            header_idx = idx
            break

    if header_idx == -1:
        raise ValueError("Could not find a valid header row containing 'term' or 'terms'.")

    # Set headers and slice
    header_row = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(df.iloc[header_idx])]
    df = df.iloc[header_idx + 1:].copy()
    df.columns = header_row

    # Dynamic column resolution
    lower_cols = {str(c).lower().strip(): c for c in header_row}

    term_col = None
    for candidate in ['term', 'terms', 'term name', 'business term']:
        if candidate in lower_cols:
            term_col = lower_cols[candidate]
            break
    if not term_col:
        for c in header_row:
            if 'term' in str(c).lower():
                term_col = c
                break

    if not term_col:
        raise ValueError(f"Could not locate a 'Term' column in headers: {header_row}")

    synonym_col = next((c for c in header_row if 'synonym' in str(c).lower()), None)
    definition_col = next((c for c in header_row if any(k in str(c).lower() for k in ['definition', 'desc', 'meaning'])), None)
    
    # Priority for logic: 'proposed logic', 'logic hint', 'dev'
    logic_col = next((c for c in header_row if 'proposed' in str(c).lower() and any(k in str(c).lower() for k in ['logic', 'hint', 'dev'])), None)
    if not logic_col:
        logic_col = next((c for c in header_row if any(k in str(c).lower() for k in ['logic', 'hint', 'sql', 'filter'])), None)

    records = []
    for _, row in df.iterrows():
        term_val = str(row[term_col]).strip() if pd.notna(row[term_col]) else ""
        if not term_val or term_val.lower() in ['nan', 'none']:
            continue

        def_val = ""
        if definition_col and pd.notna(row[definition_col]):
            val = str(row[definition_col]).strip()
            if val.lower() not in ['nan', 'none']:
                def_val = val

        syns_val = []
        if synonym_col and pd.notna(row[synonym_col]):
            raw_syn = str(row[synonym_col]).strip()
            if raw_syn.lower() not in ['nan', 'none']:
                syns_val = [s.strip() for s in raw_syn.split(',') if s.strip()]

        logic_val = ""
        if logic_col and pd.notna(row[logic_col]):
            l_val = str(row[logic_col]).strip()
            if l_val.lower() not in ['nan', 'none']:
                logic_val = l_val

        records.append({
            "term": term_val,
            "synonyms": syns_val,
            "definition": def_val,
            "logic_hint": logic_val
        })

    return records