# storage_service.py

import json
import os
import re
from datetime import datetime
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