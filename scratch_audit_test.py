import sys
import os
import io
import json
import uuid
import asyncio
import pandas as pd
from starlette.testclient import TestClient

from api_gateway import app, get_db_engine
from query_intelligence_service import (
    get_schema_retriever,
    _build_glossary_retrievers,
    parse_rag_response,
    generate_predictive_followups,
    generate_smart_suggestions,
    execute_agentic_workflow,
    PERSISTENT_GLOSSARY_DIR
)
from session_memory_store import (
    load_history_from_local_file,
    save_history_to_local_file,
    get_raw_session_history,
    remove_cancelled_interaction,
    parse_glossary_file,
    rotate_glossary_snapshots,
    get_session_lock
)
from langchain_core.messages import HumanMessage, AIMessage

client = TestClient(app)

def test_component_database():
    print("\n--- 1. Testing SQLite Database Connectivity ---")
    engine = get_db_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT COUNT(*) FROM tblCommunities;")).scalar()
        print(f"  [PASS] Database accessible. Total communities in tblCommunities: {result}")
        assert result > 0, "tblCommunities should have seeded data!"

def test_component_schema_and_glossary_retrievers():
    print("\n--- 2. Testing Vector Retrievers (Schema & Glossary) ---")
    schema_retriever = get_schema_retriever()
    docs = schema_retriever.invoke("lots and communities")
    print(f"  [PASS] Schema retriever returned {len(docs)} document chunks.")
    assert len(docs) > 0

    _build_glossary_retrievers()
    # Test async glossary retrieval via asyncio.run
    async def _test_glossary():
        from query_intelligence_service import _glossary_ensemble
        assert _glossary_ensemble is not None
        g_docs = await _glossary_ensemble.ainvoke("Spec home")
        return g_docs
    g_docs = asyncio.run(_test_glossary())
    print(f"  [PASS] Glossary ensemble retriever returned {len(g_docs)} term docs.")
    assert len(g_docs) > 0

def test_component_session_memory_and_rollback():
    print("\n--- 3. Testing Session Memory & History Rollback ---")
    test_session = f"test_session_{uuid.uuid4()}"
    
    # Save 2 turns
    msg1 = HumanMessage(content="First question")
    msg1.additional_kwargs = {"chat_id": 1, "timestamp": "2026-09-12 12:00:00"}
    msg2 = AIMessage(content="First answer")
    msg2.additional_kwargs = {"chat_id": 1, "timestamp": "2026-09-12 12:00:01"}
    
    msg3 = HumanMessage(content="Second question (uncommitted/cancelled)")
    msg3.additional_kwargs = {"chat_id": 2, "timestamp": "2026-09-12 12:01:00"}
    msg4 = AIMessage(content="Second answer")
    msg4.additional_kwargs = {"chat_id": 2, "timestamp": "2026-09-12 12:01:01"}

    save_history_to_local_file(test_session, [msg1, msg2, msg3, msg4])
    history_before = get_raw_session_history(test_session)
    assert len(history_before) == 4
    assert max(h["chat_id"] for h in history_before) == 2
    print(f"  [PASS] Saved 2 turns (4 messages).")

    # Rollback turn 2
    rolled_back = remove_cancelled_interaction(test_session)
    assert rolled_back is True
    history_after = get_raw_session_history(test_session)
    assert len(history_after) == 2
    assert max(h["chat_id"] for h in history_after) == 1
    print(f"  [PASS] Successfully rolled back uncommitted turn 2; 2 messages remain.")

    # Cleanup test file
    from session_memory_store import get_history_file_path
    filepath = get_history_file_path(test_session)
    if os.path.exists(filepath):
        os.remove(filepath)

def test_component_stream_query_endpoint():
    print("\n--- 4. Testing /api/v1/query/stream Endpoint ---")
    session_id = str(uuid.uuid4())
    
    # Conversational bypass query
    resp = client.post("/api/v1/query/stream", json={
        "session_id": session_id,
        "user_input": "Hello, what can you do?",
        "local_timestamp": "2026-09-12 19:00:00"
    })
    assert resp.status_code == 200
    lines = [json.loads(l) for l in resp.text.strip().split("\n") if l.strip()]
    assert len(lines) >= 1
    assert lines[0].get("type") == "metadata"
    print(f"  [PASS] Conversational query handled correctly: {lines[0]}")

    # Security blocked query
    resp_sec = client.post("/api/v1/query/stream", json={
        "session_id": session_id,
        "user_input": "DROP TABLE tblCommunities;",
        "local_timestamp": "2026-09-12 19:00:00"
    })
    assert resp_sec.status_code == 200
    sec_lines = [json.loads(l) for l in resp_sec.text.strip().split("\n") if l.strip()]
    assert len(sec_lines) >= 1
    # Should either be conversational NO_SQL fallback or validation failed
    print(f"  [PASS] Security query responded safely without unhandled error.")

def test_component_cancel_endpoints():
    print("\n--- 5. Testing Query Cancellation Endpoints ---")
    dummy_session = str(uuid.uuid4())
    resp1 = client.post(f"/cancel_query/{dummy_session}")
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "ignored"

    resp2 = client.post(f"/api/v1/query/cancel/{dummy_session}")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "ignored"
    print(f"  [PASS] Both cancel endpoints responded properly with expected fallback status.")

def test_component_predictive_followups():
    print("\n--- 6. Testing Predictive Next-Step Follow-ups Endpoints ---")
    session_id = str(uuid.uuid4())
    resp1 = client.post("/get_predictive_followups", json={
        "session_id": session_id,
        "user_input": "Show all lots currently in Framing status",
        "local_timestamp": "2026-09-12 19:00:00"
    })
    assert resp1.status_code == 200
    followups = resp1.json().get("followups", [])
    assert len(followups) == 4
    print(f"  [PASS] /get_predictive_followups returned 4 followups: {followups}")

    resp2 = client.post("/api/v1/query/predictive_followups", json={
        "session_id": session_id,
        "user_input": "Show all lots currently in Framing status",
        "local_timestamp": "2026-09-12 19:00:00"
    })
    assert resp2.status_code == 200
    assert len(resp2.json().get("followups", [])) == 4
    print(f"  [PASS] /api/v1/query/predictive_followups alias returned 4 followups.")

def test_component_glossary_management():
    print("\n--- 7. Testing Business Glossary Upload and Rollback ---")
    # CSV with preamble, ragged row, and dirty headers
    ragged_csv = (
        "Internal Business Glossary - Confidential\n"
        "Draft Version 2.4 - Author: Analytics\n"
        "Term Name,Synonyms,Definition,Proposed Logic\n"
        "Fast Track Lots,Rush Lots,\"Lots prioritized for completion\",\"tblLots.Status = 'Framing'\"\n"
        "Spec Mega Homes,Luxury Spec,\"Spec homes over 800k\",\"tblLots.IsSpec = 1 AND tblLots.BasePrice > 800000\"\n"
    ).encode('utf-8')

    # Upload test
    resp_upload = client.post(
        "/manage_business_glossary",
        data={"action": "upload"},
        files={"file": ("new_glossary.csv", io.BytesIO(ragged_csv), "text/csv")}
    )
    assert resp_upload.status_code == 200
    assert resp_upload.json()["status"] == "success"
    assert resp_upload.json()["total_terms"] == 2
    print(f"  [PASS] Uploaded ragged CSV successfully. Total terms: {resp_upload.json()['total_terms']}")

    # Check v1 snapshot was created by rotation
    v1_file = os.path.join(PERSISTENT_GLOSSARY_DIR, "Business_Glossary_v1.json")
    assert os.path.exists(v1_file)
    print(f"  [PASS] Glossary snapshot v1 verified at {v1_file}.")

    # Rollback to v1
    resp_rollback = client.post(
        "/manage_business_glossary",
        data={"action": "rollback", "version": "v1"}
    )
    assert resp_rollback.status_code == 200
    assert resp_rollback.json()["status"] == "success"
    print(f"  [PASS] Glossary rollback to v1 executed successfully: {resp_rollback.json()}")

    # Rollback invalid version
    resp_bad = client.post(
        "/manage_business_glossary",
        data={"action": "rollback", "version": "v99"}
    )
    assert resp_bad.status_code == 400
    print(f"  [PASS] Invalid rollback version rejected with 400 Bad Request.")

def test_component_excel_export():
    print("\n--- 8. Testing Excel Export Endpoint ---")
    session_id = str(uuid.uuid4())
    # Prepare history with a valid SQL query
    ai_content = "1. Technical Reasoning: Querying communities.\n2. Layman Explanation: Showing list.\n3. ```sql\nSELECT CommunityID, CommunityName, Region FROM tblCommunities LIMIT 5;\n```"
    msg_user = HumanMessage(content="Show communities")
    msg_user.additional_kwargs = {"chat_id": 1, "timestamp": "2026-09-12 12:00:00"}
    msg_ai = AIMessage(content=ai_content)
    msg_ai.additional_kwargs = {"chat_id": 1, "timestamp": "2026-09-12 12:00:01"}
    save_history_to_local_file(session_id, [msg_user, msg_ai])

    resp_export = client.post("/api/v1/query/export", json={
        "session_id": session_id,
        "user_input": "Export please",
        "local_timestamp": "2026-09-12 19:00:00"
    })
    assert resp_export.status_code == 200
    assert resp_export.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert len(resp_export.content) > 1000
    print(f"  [PASS] Excel export generated valid xlsx binary payload ({len(resp_export.content)} bytes).")

    # Clean up test session file
    from session_memory_store import get_history_file_path
    filepath = get_history_file_path(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == "__main__":
    print("==================================================")
    print("STARTING FULL END-TO-END APPLICATION AUDIT SUITE")
    print("==================================================")
    try:
        test_component_database()
        test_component_schema_and_glossary_retrievers()
        test_component_session_memory_and_rollback()
        test_component_stream_query_endpoint()
        test_component_cancel_endpoints()
        test_component_predictive_followups()
        test_component_glossary_management()
        test_component_excel_export()
        print("\n==================================================")
        print("ALL 8 COMPONENT TEST SUITES PASSED FLAWLESSLY!")
        print("==================================================")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
