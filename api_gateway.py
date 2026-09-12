# main.py

import io
import os
import re
import json
import asyncio
from datetime import datetime
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Generator, Any

from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from sqlalchemy import create_engine, text
from langchain_community.utilities.sql_database import SQLDatabase

# --- Local Services ---
from query_intelligence_service import (
    execute_agentic_workflow, 
    parse_rag_response, 
    generate_smart_suggestions
)
from session_memory_store import (
    load_history_from_local_file, 
    save_history_to_local_file, 
    get_raw_session_history
)

app = FastAPI(title="SQL Boomer: Making Database interactions easy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 0. CONCURRENCY & DEDICATED THREAD POOLS
# ==============================================================================
sql_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sql_worker")
excel_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="excel_worker")
llm_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="llm_worker")

MAX_EXCEL_ROWS = 150000 
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "sqlite:///enterprise_mock.db")

# ==============================================================================
# 1. DATABASE CONNECTION & LAZY SQL DATABASE WRAPPER
# ==============================================================================
class LazySQLDatabase(SQLDatabase):
    """
    Lightweight SQLDatabase wrapper that bypasses eager table metadata reflection.
    Prevents startup hangs or crashes on legacy schemas with complex constraints.
    """
    def __init__(self, engine):
        self._engine = engine
        self._schema = None
        self._metadata = None
        self.include_tables = []
        self.ignore_tables = []
        self._sample_rows_in_table_info = 0

    @property
    def engine(self):
        return self._engine

_global_engine = None
_global_lazy_db = None

def get_db_engine():
    """Returns a globally pooled SQLAlchemy engine for SQLite."""
    global _global_engine
    if _global_engine is None:
        _global_engine = create_engine(SQLITE_DB_PATH, pool_size=5, max_overflow=10)
    return _global_engine

def get_db_connection() -> LazySQLDatabase:
    """Returns a lazy SQLDatabase wrapper bypassing expensive startup reflection."""
    global _global_lazy_db
    if _global_lazy_db is None:
        engine = get_db_engine()
        _global_lazy_db = LazySQLDatabase(engine)
    return _global_lazy_db

# ==============================================================================
# 2. PYDANTIC MODELS
# ==============================================================================
class QueryRequest(BaseModel):
    session_id: UUID = Body(default_factory=uuid4)
    user_input: str = Body(...)
    # The frontend UI will send its local time here (e.g., "2026-03-15 12:51:39")
    local_timestamp: str | None = Body(default=None)

class QuerySummaryResponse(BaseModel):
    type: str = "metadata"
    session_id: UUID
    tech_summary: str 
    gen_summary: str   
    raw_sql: str | None = None
    error: str | None = None
    message: str | None = None

# ==============================================================================
# 3. STRICT SECURITY & VALIDATION (Feature 5)
# ==============================================================================
def _validate_sql_security(query: str) -> str | None:
    """Strips comments and explicitly blocks dangerous SQL operations."""
    clean_query = re.sub(r'--.*?\n|/\*.*?\*/', '', query + '\n', flags=re.DOTALL).strip().lower()

    if not clean_query.startswith("select") and not clean_query.startswith("with"):
        return "Security Error: Only SELECT queries are permitted."
    
    # Pragma added for SQLite to prevent config tampering
    forbidden_keywords = ["drop", "delete", "update", "insert", "alter", "exec", "truncate", "grant", "revoke", "pragma"]
    
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', clean_query):
            return f"Security Error: Query contains forbidden keyword '{kw}'."
            
    return None

def _get_friendly_error(error_type: str, details: str) -> str:
    """Deterministic Error Handling (Replaces Feature 3 LLM Chain)."""
    if error_type == "connection_error":
        return "We are experiencing temporary database connection issues. Please try again."
    elif error_type == "execution_error":
        return "The database couldn't process this request. The specific data might be missing or formatted differently."
    else: 
        return "I couldn't safely generate a query for this request. Try rephrasing with specific business terms."

# ==============================================================================
# 4. ASYNC ITERATORS & EXECUTION
# ==============================================================================
def _next_chunk_safe(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

class AsyncPandasIterator:
    """Wraps a Pandas chunk iterator to prevent blocking the FastAPI event loop."""
    def __init__(self, sync_iterator, executor):
        self.sync_iterator = sync_iterator
        self.executor = executor
        self.loop = asyncio.get_event_loop()

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self.loop.run_in_executor(self.executor, _next_chunk_safe, self.sync_iterator)
        if data is None:
            raise StopAsyncIteration
        return data

async def _execute_sql_generator(engine, query: str, chunk_size: int = 500) -> tuple[Any, str | None]:
    loop = asyncio.get_event_loop()
    
    def fetch():
        error = _validate_sql_security(query)
        if error: return None, error
        try:
            return pd.read_sql_query(query, engine, chunksize=chunk_size), None
        except Exception as e:
            return None, str(e)

    iterator, error = await loop.run_in_executor(sql_executor, fetch)
    if error: return None, error
    return AsyncPandasIterator(iterator, sql_executor), None

# ==============================================================================
# 5. DATA SANITIZATION (Features 8 & 9)
# ==============================================================================
def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes NaN/NaT and strips illegal XML characters to prevent crashes."""
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    
    illegal_char_pattern = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]')
    str_cols = df.select_dtypes(include=['object']).columns
    
    for col in str_cols:
        df[col] = df[col].astype(str).str.replace(illegal_char_pattern, '', regex=True)
    return df

def _sanitize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Prevents empty or duplicate headers from breaking xlsxwriter."""
    new_cols = []
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        if not col_str:
            col_str = f"Column_{i+1}"
        new_cols.append(col_str)
    df.columns = new_cols
    return df

# ==============================================================================
# 6. SMART EXCEL EXPORT (Features 11, 12, 13)
# ==============================================================================
def _add_smart_chart(writer, df, sheet_name, raw_sql: str):
    """Detects X-axis/Y-axis based on dtypes and injects native Excel charts."""
    is_aggregated = any(k in raw_sql.upper() for k in ["GROUP BY", "SUM(", "COUNT(", "AVG("])
    if not is_aggregated or df.empty or len(df.columns) < 2 or len(df) > 1000:
        return

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    max_row, max_col = df.shape

    # Find Categories (X-Axis): Text/Date columns before the first number
    cat_col_end_idx = 0
    for i, col in enumerate(df.columns):
        if is_numeric_dtype(df[col]):
            break
        cat_col_end_idx = i 
    
    if is_numeric_dtype(df.iloc[:, 0]): cat_col_end_idx = 0

    # Find Values (Y-Axis)
    y_col_indices = [i for i, col in enumerate(df.columns) if i > cat_col_end_idx and is_numeric_dtype(df[col])]
    if not y_col_indices: return 

    x_col_data = df.iloc[:, 0]
    is_time_col = is_datetime64_any_dtype(x_col_data) or 'date' in str(df.columns[0]).lower()
    chart_type = 'line' if is_time_col else 'column'

    chart = workbook.add_chart({'type': chart_type})
    for col_idx in y_col_indices:
        chart.add_series({
            'name':       [sheet_name, 0, col_idx], 
            'categories': [sheet_name, 1, 0, max_row, cat_col_end_idx], 
            'values':     [sheet_name, 1, col_idx, max_row, col_idx], 
        })

    chart.set_style(10)
    chart.set_size({'width': 900, 'height': 500}) 
    worksheet.insert_chart(1, max_col + 2, chart)

def _execute_excel_worker(engine, query: str) -> tuple[bytes | None, str, str | None]:
    error = _validate_sql_security(query)
    if error: return None, "", error

    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return None, "", "No data found."

        # Feature 11: 150k CSV Fallback
        if len(df) > MAX_EXCEL_ROWS:
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            return csv_buffer.getvalue(), "csv", None

        # Feature 12 & 13: Excel Styling & Charting
        df = _sanitize_headers(_sanitize_dataframe(df))
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Query Results', startrow=1, header=False, index=False)
            worksheet = writer.sheets['Query Results']
            max_row, max_col = df.shape
            
            worksheet.add_table(0, 0, max_row, max_col - 1, {
                'columns': [{'header': col} for col in df.columns],
                'style': 'TableStyleMedium9'
            })
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, max_col - 1, 20)
            
            _add_smart_chart(writer, df, 'Query Results', query)

        return excel_buffer.getvalue(), "xlsx", None

    except Exception as e:
        return None, "", str(e)

# ==============================================================================
# 7. ENDPOINTS
# ==============================================================================
@app.post("/api/v1/query/stream")
async def execute_and_summarize(request_data: QueryRequest):
    session_id_str = str(request_data.session_id)
    user_input = request_data.user_input.strip()

    # Fallback to server time if the frontend forgets to send it
    client_time = request_data.local_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if len(user_input) < 3:
        raise HTTPException(400, "Please provide a valid question.")

    engine = get_db_engine()
    history = load_history_from_local_file(session_id_str)
    
    try:
        rag_answer, updated_history = await execute_agentic_workflow(user_input, history, client_time)
        save_history_to_local_file(session_id_str, updated_history)
    except Exception as e:
        return QuerySummaryResponse(
            session_id=request_data.session_id, tech_summary="API Error", gen_summary="System error.",
            error=str(e), message=_get_friendly_error("connection_error", str(e))
        )

    raw_sql, tech_summary, gen_summary = parse_rag_response(rag_answer)

    # ---> The SQL Formatter Safety Net <---
    if raw_sql and raw_sql != "NO_SQL":
        # Force a space between any alphanumeric character/quote and a major SQL keyword.
        # Strict capitalization prevents splitting columns like "ActiveFrom"
        raw_sql = re.sub(
            r'([a-zA-Z0-9_\]\'\"])(SELECT|FROM|WHERE|JOIN|INNER|LEFT|RIGHT|GROUP BY|ORDER BY|LIMIT|HAVING)\b', 
            r'\1 \n\2', 
            raw_sql 
        )
    # --------------------------------------------------------

    # Conversational Bypass
    if raw_sql and "NO_SQL" in raw_sql:
        async def conv_stream():
            yield json.dumps({"type": "metadata", "session_id": session_id_str, "tech_summary": "Chat", "gen_summary": gen_summary}) + "\n"
        return StreamingResponse(conv_stream(), media_type="application/x-ndjson")

    is_valid = raw_sql and (raw_sql.lower().startswith("select") or raw_sql.lower().startswith("with"))
    if not is_valid:
        return QuerySummaryResponse(
            session_id=request_data.session_id, tech_summary=tech_summary, gen_summary=gen_summary,
            error="SQL Validation Failed", message=_get_friendly_error("generation_failure", "")
        )

    db_iterator, exec_error = await _execute_sql_generator(engine, raw_sql)

    if exec_error:
        return QuerySummaryResponse(
            session_id=request_data.session_id, tech_summary=tech_summary, gen_summary="DB Error",
            error=exec_error, message=_get_friendly_error("execution_error", "")
        )

    # Feature 14: Graceful handling of Zero Rows
    first_chunk, is_empty = None, False
    try:
        first_chunk = await db_iterator.__anext__()
        if first_chunk.empty: is_empty = True
    except StopAsyncIteration:
        is_empty = True

    if is_empty:
        async def empty_stream():
            yield json.dumps({
                "type": "metadata", "session_id": session_id_str,
                "tech_summary": tech_summary, 
                "gen_summary": gen_summary + "\n\n**Note:** Zero records matched this query.",
                "raw_sql": raw_sql, "error": None
            }) + "\n"
        return StreamingResponse(empty_stream(), media_type="application/x-ndjson")

    async def full_stream():
        yield json.dumps({"type": "metadata", "session_id": session_id_str, "tech_summary": tech_summary, "gen_summary": gen_summary, "raw_sql": raw_sql}) + "\n"
        
        chunk = _sanitize_dataframe(first_chunk)
        for row in chunk.to_dict('records'): yield json.dumps(row, default=str) + "\n"
        
        async for chunk_df in db_iterator:
            chunk_df = _sanitize_dataframe(chunk_df)
            for row in chunk_df.to_dict('records'):
                yield json.dumps(row, default=str) + "\n"

    return StreamingResponse(full_stream(), media_type="application/x-ndjson")

@app.post("/api/v1/query/export")
async def execute_and_return_excel(request_data: QueryRequest):
    session_id_str = str(request_data.session_id)
    history = load_history_from_local_file(session_id_str)
    
    if not history or not hasattr(history[-1], 'content'):
        raise HTTPException(400, "No active valid history to export.")

    raw_sql, _, _ = parse_rag_response(history[-1].content)
    if not raw_sql or "NO_SQL" in raw_sql:
        raise HTTPException(400, "Previous query is conversational and cannot be exported.")

    loop = asyncio.get_event_loop()
    file_content, ext, error = await loop.run_in_executor(excel_executor, _execute_excel_worker, get_db_engine(), raw_sql)
    
    if error: raise HTTPException(400, _get_friendly_error("execution_error", error))

    # 1. Grab the client's exact time (or fallback to server time if missing)
    client_time_raw = request_data.local_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Sanitize the string for the OS (e.g., "2026-03-15 08:22:16" -> "20260315_082216")
    safe_timestamp = client_time_raw.replace("-", "").replace(":", "").replace(" ", "_")

    media_type = "text/csv" if ext == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    filename = f"Data_Export_{safe_timestamp}.{ext}"

    return Response(content=file_content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.get("/api/v1/session/{session_id}/suggestions")
async def get_smart_suggestions_endpoint(session_id: UUID):
    raw_history = get_raw_session_history(str(session_id))
    loop = asyncio.get_event_loop()
    suggestions = await loop.run_in_executor(sql_executor, lambda: asyncio.run(generate_smart_suggestions(raw_history)))
    return {"suggestions": suggestions}