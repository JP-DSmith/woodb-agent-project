"""Supplier document records: format checks, and how documents change (or don't change) answers."""
import pytest

from mcp_servers import cusma_tools as tools
from mcp_servers import document_records as docs

BASE = {"doc_id": "DOC-9001", "received_date": "2026-10-01", "doc_type": "invoice", "language": "en",
        "supplier_id": "SUP-015", "part_description": "SBR / BR synthetic rubber", "hs_code_shown": "4002.19",
        "origin_shown": "US", "status": "pending", "defect_ids": "", "checked_by": "intake_agent"}


@pytest.fixture
def records(tmp_path, monkeypatch):
    path = tmp_path / "document_records.csv"
    monkeypatch.setattr(docs, "RECORDS_FILE", path)
    return path


def add(**changes):
    out = docs.append_record({**BASE, **changes})
    assert out["ok"], out


def test_no_records_file_changes_nothing(records):
    assert tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")["data"]["verdict"] == "qualifies"
    assert "documents" not in tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")["data"]


def test_bad_records_are_refused(records):
    bad = docs.append_record({**BASE, "status": "approved", "received_date": "Oct 1"})
    assert not bad["ok"] and "status" in bad["error"]["message"] and "received_date" in bad["error"]["message"]
    assert not docs.append_record({**BASE, "status": "flagged"})["ok"]          # flagged needs a defect
    assert "X9" in docs.append_record({**BASE, "status": "flagged", "defect_ids": "C6;X9"})["error"]["message"]


def test_minor_defect_confirmed_by_a_person_changes_nothing(records):
    add(status="verified", defect_ids="O2", checked_by="person")             # old address: record with a note
    assert tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")["data"]["verdict"] == "qualifies"


def test_answer_keys_can_score_records(records):
    """The answer key's defect_id values go straight into defect_ids."""
    key = [{"defect_id": "C6"}, {"defect_id": "T1"}]
    add(status="flagged", defect_ids=";".join(d["defect_id"] for d in key))
    assert docs.defect_list(docs.load_records()[0]) == ["C6", "T1"]


def test_pending_document_is_a_note_only(records):
    add()
    r = tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")
    assert r["data"]["verdict"] == "qualifies" and r["data"]["documents"][0]["doc_id"] == "DOC-9001"


def test_flagged_document_moves_qualifies_to_needs_review(records):
    add(status="flagged", defect_ids="C6", origin_shown="CN")
    r = tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")
    assert r["data"]["verdict"] == "needs_review" and r["data"]["verdict_before_documents"] == "qualifies"
    assert any("DOC-9001" in w for w in r["warnings"])


def test_document_received_after_the_date_is_ignored(records):
    add(status="flagged", defect_ids="C1", received_date="2026-12-01")
    assert tools.call_tool("check_origin", {"sku": "NB-0101", "as_of": "2026-11-06"}, "A")["data"]["verdict"] == "qualifies"


def test_documents_never_make_a_boot_qualify(records):
    failing = next(s for s in tools.eng.SKUS
                   if tools.call_tool("check_origin", {"sku": s}, "A")["data"]["verdict"] == "does_not_qualify")
    sup = tools._boot_suppliers(failing, tools._as_of(None))[0]
    add(supplier_id=sup, status="verified", doc_type="origin_declaration", defect_ids="", checked_by="person")
    add(doc_id="DOC-9002", supplier_id=sup, status="flagged", defect_ids="T1")
    assert tools.call_tool("check_origin", {"sku": failing}, "A")["data"]["verdict"] == "does_not_qualify"


def test_rejected_and_clean_verified_documents_do_not_change_the_verdict(records):
    add(status="rejected", defect_ids="T1;C6", checked_by="person")
    add(doc_id="DOC-9002", status="verified", doc_type="origin_declaration", checked_by="person")
    assert tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")["data"]["verdict"] == "qualifies"


def test_confirmed_problem_keeps_needs_review(records):
    add(status="verified", defect_ids="O1", checked_by="person")
    assert tools.call_tool("check_origin", {"sku": "NB-0101"}, "A")["data"]["verdict"] == "needs_review"


def test_flagged_document_blocks_certificate_and_shows_duty_both_ways(records):
    po = next(p for p, o in tools.ORDERS.items()
              if tools.call_tool("draft_certification", {"po_number": p}, "A")["ok"])
    sku = tools.ORDERS[po]["sku"]
    sup = tools._boot_suppliers(sku, tools._as_of(tools.ORDERS[po]["ship_date"]))[0]
    add(supplier_id=sup, status="flagged", defect_ids="M4")
    assert tools.call_tool("draft_certification", {"po_number": po}, "A")["error"]["code"] == "NOT_SUPPORTED"
    d = tools.call_tool("order_duty", {"po_number": po}, "A")["data"]
    assert d["verdict"] == "needs_review" and d["duty_payable_usd"] == d["duty_if_not_qualifying_usd"]


def test_materials_show_documents_per_line_and_tier_c_sees_no_supplier(records):
    add(status="flagged", defect_ids="C6")
    a = tools.call_tool("get_materials", {"sku": "NB-0101"}, "A")["data"]
    assert next(l for l in a if l["supplier_id"] == "SUP-015")["documents"][0]["doc_id"] == "DOC-9001"
    c = tools.call_tool("check_origin", {"sku": "NB-0101"}, "C")
    assert c["data"]["verdict"] == "needs_review"
    assert "SUP-015" not in str(c) and "supplier_id" not in str(c["data"])
