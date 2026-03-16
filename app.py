"""
Intelligent Expense & Invoice Automation System
================================================
A unified Streamlit demo showcasing two enterprise automation scenarios:
  UC1 — T&E (Travel & Entertainment) Employee Expense Processing
  UC2 — Accounts Payable Vendor Invoice Processing

Both use cases share a common sidebar navigation and leverage:
  - Google Cloud Document AI  → receipt / invoice entity extraction
  - Vertex AI Gemini 2.5 Flash → policy evaluation, anomaly detection, comms drafting
  - st.session_state → persistent in-memory demo state (no database required)
"""

import os
import json
import uuid
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd

# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="AI Expense & Invoice System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS                                                 ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding-top: 1rem; max-width: 1200px; }
.main { background: #f0f2f6; }

/* ── Hero banners ── */
.hero {
    background: linear-gradient(135deg, #1e293b 0%, #334155 60%, #475569 100%);
    color: white; padding: 1.6rem 2.2rem; border-radius: 14px; margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
    border-left: 5px solid #6366f1;
}
.hero-ap {
    background: linear-gradient(135deg, #1e293b 0%, #334155 60%, #475569 100%);
    color: white; padding: 1.6rem 2.2rem; border-radius: 14px; margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
    border-left: 5px solid #06b6d4;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -5%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-ap::before {
    content: ''; position: absolute; top: -50%; right: -5%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1, .hero-ap h1 {
    margin: 0 0 0.35rem; font-size: 1.4rem; font-weight: 700;
    position: relative; z-index: 1; letter-spacing: -0.01em;
}
.hero p, .hero-ap p {
    margin: 0; opacity: 0.7; font-size: 0.85rem;
    position: relative; z-index: 1;
}

/* ── Stat cards ── */
.stat-card {
    background: white; border-radius: 12px;
    padding: 1rem 1.1rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: none; transition: transform 0.15s, box-shadow 0.15s;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.stat-card .num {
    font-size: 1.65rem; font-weight: 700; margin: 0;
    letter-spacing: -0.02em;
}
.stat-card .lbl {
    font-size: 0.72rem; color: #94a3b8; margin: 0.2rem 0 0;
    text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;
}

/* ── Info block ── */
.info-block {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem; margin-bottom: 0.75rem; line-height: 1.7;
    color: #1e293b;
}
.info-block-ap {
    background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
    border-left: 4px solid #06b6d4;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem; margin-bottom: 0.75rem; line-height: 1.7;
    color: #1e293b;
}

/* ── Badges ── */
.badge {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase;
}
.b-pending  { background: #fef3c7; color: #92400e; }
.b-approved { background: #d1fae5; color: #065f46; }
.b-rejected { background: #fee2e2; color: #991b1b; }
.b-auto     { background: #dbeafe; color: #1e40af; }
.b-flagged  { background: #fce7f3; color: #9d174d; }
.b-hold     { background: #e0e7ff; color: #3730a3; }
.b-paid     { background: #d1fae5; color: #065f46; }
.b-mismatch { background: #fef3c7; color: #92400e; }

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #1e293b;
}
div[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
div[data-testid="stSidebar"] .stRadio label span {
    color: #cbd5e1 !important;
}
div[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}

/* ── Placeholder boxes ── */
.placeholder-box {
    background: #f1f5f9; border: 2px dashed #cbd5e1; border-radius: 12px;
    padding: 2.5rem 1rem; text-align: center; color: #94a3b8;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 2.5rem 1rem; color: #94a3b8;
    background: white; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  GCP CLIENT HELPERS (cached for session lifetime)           ║
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_resource
def get_docai_client():
    """
    Initialize Document AI client with regional endpoint.
    ── REAL GCP CALL ──
    Requires: GOOGLE_APPLICATION_CREDENTIALS or ADC login,
              env vars PROJECT_ID, LOCATION, PROCESSOR_ID.
    """
    from google.cloud import documentai_v1 as documentai
    project_id = os.environ.get("PROJECT_ID", "mcps-owntest")
    location = os.environ.get("LOCATION", "us")
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    return client, project_id, location


@st.cache_resource
def get_gemini_model():
    """
    Initialize Vertex AI Gemini 2.5 Flash.
    ── REAL GCP CALL ──
    Requires: ADC credentials with Vertex AI User role on the project.
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel
    project_id = os.environ.get("PROJECT_ID", "mcps-owntest")
    vertexai.init(project=project_id, location="us-central1")
    return GenerativeModel("gemini-2.5-flash")


# ╔══════════════════════════════════════════════════════════════╗
# ║  UTILITY FUNCTIONS                                          ║
# ╚══════════════════════════════════════════════════════════════╝
def extract_with_docai(file_bytes: bytes, mime_type: str) -> dict:
    """
    Call Document AI to extract entities from a document.
    ── REAL GCP CALL ──
    Returns dict with:
      - Top-level fields: entity_type -> mention_text (e.g., total_amount, invoice_date)
      - "_line_items": list of dicts, each with sub-entity fields
        (e.g., line_item/description, line_item/quantity, line_item/unit_price, line_item/amount)
      - "_full_text": first 2000 chars of OCR text
    """
    from google.cloud import documentai_v1 as documentai
    client, project_id, location = get_docai_client()
    processor_id = os.environ.get("PROCESSOR_ID", "ca7b526548cdf4ef")
    resource_name = client.processor_path(project_id, location, processor_id)
    req = documentai.ProcessRequest(
        name=resource_name,
        raw_document=documentai.RawDocument(content=file_bytes, mime_type=mime_type),
    )
    result = client.process_document(request=req)

    extracted = {}
    line_items = []

    for entity in result.document.entities:
        # Line items are parent entities with nested sub-entities (properties)
        if entity.type_ == "line_item":
            row = {}
            for prop in entity.properties:
                # Sub-entity types: line_item/description, line_item/quantity,
                # line_item/unit_price, line_item/amount, etc.
                short_key = prop.type_.replace("line_item/", "")
                row[short_key] = prop.mention_text
            line_items.append(row)
        else:
            extracted[entity.type_] = entity.mention_text

    extracted["_line_items"] = line_items
    extracted["_full_text"] = (result.document.text or "")[:2000]
    return extracted


def call_gemini(prompt: str) -> str:
    """
    Send prompt to Gemini 2.5 Flash via Vertex AI and return text.
    ── REAL GCP CALL ──
    """
    model = get_gemini_model()
    return model.generate_content(prompt).text


def get_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "pdf": "application/pdf", "png": "image/png",
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "tif": "image/tiff", "tiff": "image/tiff",
    }.get(ext, "application/pdf")


def gen_id(prefix: str = "EXP") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"


def badge(status: str) -> str:
    css_map = {
        "Pending Approval": "b-pending", "Approved": "b-approved",
        "Rejected": "b-rejected", "Auto-Approved": "b-auto",
        "Flagged - Justification Required": "b-flagged",
        "Pending Review": "b-pending", "Approved for Payment": "b-paid",
        "On Hold": "b-hold", "Flagged - Amount Mismatch": "b-mismatch",
    }
    css = css_map.get(status, "b-pending")
    return f'<span class="badge {css}">{status}</span>'


def stat_card(value: str, label: str, color: str = "#6366f1") -> str:
    return (
        f'<div class="stat-card">'
        f'<p class="num" style="color:{color}">{value}</p>'
        f'<p class="lbl">{label}</p></div>'
    )


def parse_amount(raw: str) -> float:
    """Best-effort parse of a currency string to float."""
    try:
        return float(str(raw).replace(",", "").replace("$", "").replace("NT", "").strip())
    except (ValueError, AttributeError):
        return 0.0


# ── Status color mapping for styled dataframes ──
STATUS_COLORS = {
    # UC1
    "Auto-Approved":  {"bg": "#dbeafe", "fg": "#1e40af"},
    "Approved":       {"bg": "#d1fae5", "fg": "#065f46"},
    "Pending Approval": {"bg": "#fef3c7", "fg": "#92400e"},
    "Rejected":       {"bg": "#fee2e2", "fg": "#991b1b"},
    "Flagged - Justification Required": {"bg": "#fce7f3", "fg": "#9d174d"},
    # UC2
    "Pending Review":          {"bg": "#fef3c7", "fg": "#92400e"},
    "Approved for Payment":    {"bg": "#d1fae5", "fg": "#065f46"},
    "On Hold":                 {"bg": "#e0e7ff", "fg": "#3730a3"},
    "Flagged - Amount Mismatch": {"bg": "#fce7f3", "fg": "#9d174d"},
    # PO
    "Open":   {"bg": "#dbeafe", "fg": "#1e40af"},
    "Closed": {"bg": "#f1f5f9", "fg": "#475569"},
}


def style_status_col(df: pd.DataFrame):
    """Apply colored background to the 'Status' column if present."""
    def _color_status(val):
        c = STATUS_COLORS.get(val, {"bg": "#f1f5f9", "fg": "#475569"})
        return (
            f"background-color: {c['bg']}; color: {c['fg']}; "
            f"font-weight: 600; border-radius: 4px; text-align: center;"
        )
    styler = df.style
    if "Status" in df.columns:
        styler = styler.map(_color_status, subset=["Status"])
    return styler


# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE — UC1 (T&E)                                  ║
# ╚══════════════════════════════════════════════════════════════╝
if "te_policies" not in st.session_state:
    st.session_state.te_policies = {
        "auto_approve_limit": 50.0,
        "flag_threshold": 200.0,
    }

if "te_expenses" not in st.session_state:
    st.session_state.te_expenses = [
        {
            "id": "EXP-A1B2C3", "employee": "Alice Chen", "amount": 42.50,
            "date": "2026-03-10", "category": "Meals & Entertainment",
            "description": "Team lunch — quarterly planning",
            "status": "Auto-Approved", "justification": "",
            "auditor_notes": "", "follow_up": None,
            "image_bytes": None, "mime_type": None, "messages": [],
        },
        {
            "id": "EXP-D4E5F6", "employee": "Bob Wang", "amount": 350.00,
            "date": "2026-03-11", "category": "Travel & Transportation",
            "description": "Client visit — Taipei round-trip HSR",
            "status": "Pending Approval",
            "justification": "Client onsite meeting for Project Falcon. Pre-approved by PM.",
            "auditor_notes": "", "follow_up": None,
            "image_bytes": None, "mime_type": None, "messages": [],
        },
        {
            "id": "EXP-G7H8I9", "employee": "Carol Liu", "amount": 1280.00,
            "date": "2026-03-08", "category": "Equipment & Supplies",
            "description": "Ergonomic keyboard + monitor stand",
            "status": "Pending Approval",
            "justification": "RSI prevention — doctor's recommendation attached.",
            "auditor_notes": "", "follow_up": None,
            "image_bytes": None, "mime_type": None, "messages": [],
        },
    ]


# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE — UC2 (Accounts Payable)                     ║
# ╚══════════════════════════════════════════════════════════════╝
if "ap_po_database" not in st.session_state:
    # ── The "perfect" approved PO — the single source of truth for 3-way matching ──
    st.session_state.ap_po_database = {
        "PO-2026-001": {
            "vendor": "GlobalTech Solutions", "description": "IT Equipment Refresh — Q1 2026",
            "total": 11750.00, "status": "Open",
            "line_items": [
                {"item": "ThinkPad Laptop", "qty": 10, "unit_price": 1000.00},
                {"item": "27-inch Monitor", "qty": 5, "unit_price": 200.00},
                {"item": "Wireless Mouse", "qty": 15, "unit_price": 50.00},
            ],
        },
    }

if "ap_settings" not in st.session_state:
    st.session_state.ap_settings = {
        "price_tolerance_pct": 5.0,
    }

if "ap_invoices" not in st.session_state:
    # ── The vendor's invoice — deliberately contains 3 types of discrepancies ──
    # 1. PRICING: ThinkPad billed at $1,050 vs PO approved $1,000 (+$50/unit)
    # 2. QUANTITY: 27-inch Monitors — billed 7 units vs PO approved 5 (+2 extra)
    # 3. HIDDEN FEE: "Express Shipping & Handling" $300 — not in the PO at all
    st.session_state.ap_invoices = [
        {
            "id": "INV-2026-042", "vendor": "GlobalTech Solutions",
            "invoice_number": "GT-2026-1887",
            "amount": 13150.00, "currency": "USD",
            "date": "2026-03-10", "due_date": "2026-04-09",
            "po_number": "PO-2026-001",
            "category": "IT Hardware", "payment_terms": "Net 30",
            "status": "Pending Review",
            "line_items": [
                {"description": "ThinkPad Laptop", "qty": 10, "unit_price": 1050.00, "total": 10500.00},
                {"description": "27-inch Monitor", "qty": 7, "unit_price": 200.00, "total": 1400.00},
                {"description": "Wireless Mouse", "qty": 15, "unit_price": 50.00, "total": 750.00},
                {"description": "Express Shipping & Handling", "qty": 1, "unit_price": 300.00, "total": 300.00},
            ],
            "notes": "",
            "image_bytes": None, "mime_type": None,
        },
    ]


# ╔══════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — Master Navigation                                ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.5rem 0 0.3rem;'>"
        "<span style='font-size:1.6rem;'>🧾</span><br/>"
        "<span style='font-size:1rem;font-weight:700;letter-spacing:-0.02em;'>"
        "ExpenseAI</span><br/>"
        "<span style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;"
        "color:#64748b !important;'>Intelligent Automation</span>"
        "</div>", unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
                "color:#64748b !important;margin-bottom:0.3rem;'>Scenario</p>",
                unsafe_allow_html=True)
    scenario = st.radio(
        "Select Scenario",
        ["Use Case 1: T&E (Employee Expenses)",
         "Use Case 2: Accounts Payable (Vendor Invoices)"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
                "color:#64748b !important;margin-bottom:0.3rem;'>Role</p>",
                unsafe_allow_html=True)
    role = st.radio(
        "Select Role",
        ["Submitter (Employee/Vendor)", "Reviewer (Auditor/AP Specialist)"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Context panel ──
    if "Use Case 1" in scenario:
        p = st.session_state.te_policies
        st.markdown(
            "<div style='background:#334155;border-radius:8px;padding:0.8rem 1rem;'>"
            "<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;"
            "color:#94a3b8 !important;margin:0 0 0.5rem;'>Active Policies</p>"
            f"<p style='margin:0 0 0.25rem;font-size:0.82rem;'>Auto-approve &le; <strong>${p['auto_approve_limit']:.0f}</strong></p>"
            f"<p style='margin:0;font-size:0.82rem;'>Flag threshold &gt; <strong>${p['flag_threshold']:.0f}</strong></p>"
            "</div>", unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#334155;border-radius:8px;padding:0.8rem 1rem;'>"
            "<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;"
            "color:#94a3b8 !important;margin:0 0 0.5rem;'>AP Settings</p>"
            f"<p style='margin:0 0 0.25rem;font-size:0.82rem;'>Tolerance: <strong>{st.session_state.ap_settings['price_tolerance_pct']:.1f}%</strong></p>"
            f"<p style='margin:0;font-size:0.82rem;'>POs in system: <strong>{len(st.session_state.ap_po_database)}</strong></p>"
            "</div>", unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<div style='text-align:center;padding:0.5rem 0;'>"
        "<p style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#475569 !important;margin:0 0 0.3rem;'>Powered by</p>"
        "<p style='font-size:0.72rem;color:#94a3b8 !important;margin:0;line-height:1.5;'>"
        "Document AI<br/>Gemini 2.5 Flash<br/>Streamlit</p>"
        "</div>", unsafe_allow_html=True,
    )

is_uc1 = "Use Case 1" in scenario
is_submitter = "Submitter" in role


# ╔══════════════════════════════════════════════════════════════╗
# ║                                                              ║
# ║  USE CASE 1 — T&E  (Employee Expenses)                      ║
# ║                                                              ║
# ╚══════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────
# UC1 — SUBMITTER (Employee)
# ──────────────────────────────────────────────────────────────
def uc1_submitter():
    expenses = st.session_state.te_expenses
    policies = st.session_state.te_policies

    st.markdown(
        '<div class="hero">'
        "<h1>💳 Employee Expense Portal</h1>"
        "<p>Submit receipts, track approvals, and let AI speed up your reimbursements.</p>"
        "</div>", unsafe_allow_html=True,
    )

    tab_submit, tab_history = st.tabs(["📤 Submit New Expense", "📜 My Expense History"])

    # ── History Tab ──────────────────────────────
    with tab_history:
        st.markdown("### 📜 My Expense History")
        if not expenses:
            st.info("No expenses submitted yet.")
        else:
            df = pd.DataFrame([{
                "ID": e["id"], "Amount": f"${e['amount']:,.2f}", "Date": e["date"],
                "Category": e["category"], "Description": e["description"],
                "Status": e["status"],
            } for e in expenses])
            st.dataframe(style_status_col(df), width="stretch", hide_index=True)

    # ── Submit Tab ───────────────────────────────
    with tab_submit:
        st.markdown("### 📤 Submit New Expense")

        # Step 1: Upload
        st.markdown("**Step 1:** Upload your receipt")
        uploaded = st.file_uploader(
            "Drop receipt here (PNG, JPG, PDF, TIFF)",
            type=["png", "jpg", "jpeg", "pdf", "tif", "tiff"],
            key="uc1_upload",
        )

        if uploaded is None:
            return

        file_bytes = uploaded.read()
        mime_type = get_mime(uploaded.name)

        # Step 2: AI Extraction
        st.divider()
        st.markdown("**Step 2:** AI Extraction")

        col_img, col_data = st.columns([1, 1.5])
        with col_img:
            if mime_type.startswith("image"):
                st.image(file_bytes, width="stretch", caption="Uploaded Receipt")
            else:
                st.info("📄 PDF uploaded — preview not available")

        with col_data:
            # ── REAL Document AI call ──
            with st.spinner("🔍 Document AI is extracting data..."):
                try:
                    extracted = extract_with_docai(file_bytes, mime_type)
                    raw_amount = extracted.get("total_amount", extracted.get("net_amount", ""))
                    raw_date = extracted.get("receipt_date", extracted.get("invoice_date", ""))
                    raw_supplier = extracted.get("supplier_name", extracted.get("supplier_address", ""))
                except Exception as e:
                    st.error(f"Document AI error: {e}")
                    raw_amount, raw_date, raw_supplier = "", "", ""
                    extracted = {}

            amount_val = parse_amount(raw_amount)

            st.markdown("##### ✏️ Verify & Edit Extracted Data")
            fc1, fc2 = st.columns(2)
            with fc1:
                amount_input = st.number_input(
                    "💰 Amount ($)", value=amount_val, min_value=0.0, step=1.0, key="uc1_amt",
                )
            with fc2:
                date_input = st.text_input("📅 Date", value=raw_date or str(date.today()), key="uc1_date")

            fc3, fc4 = st.columns(2)
            with fc3:
                supplier_input = st.text_input("🏪 Supplier", value=raw_supplier or "", key="uc1_sup")
            with fc4:
                category = st.selectbox(
                    "🏷️ Category",
                    ["Meals & Entertainment", "Travel & Transportation",
                     "Equipment & Supplies", "Software & Subscriptions",
                     "Training & Education", "Other"],
                    key="uc1_cat",
                )

            description = st.text_input(
                "📝 Brief Description",
                placeholder="e.g., Team lunch for Q1 kickoff",
                key="uc1_desc",
            )

        # Step 3: Policy Check & Submit
        st.divider()
        st.markdown("**Step 3:** Policy Check & Submit")

        auto_limit = policies["auto_approve_limit"]
        flag_limit = policies["flag_threshold"]
        needs_justification = amount_input > flag_limit
        is_auto = amount_input <= auto_limit

        if is_auto:
            st.success(
                f"✅ **Auto-Approve Eligible** — ${amount_input:,.2f} is within the "
                f"auto-approve limit (${auto_limit:,.0f})."
            )
        elif needs_justification:
            st.warning(
                f"⚠️ **Justification Required** — ${amount_input:,.2f} exceeds the "
                f"policy threshold (${flag_limit:,.0f}). Provide a business justification below."
            )
        else:
            st.info(
                f"📋 **Standard Review** — ${amount_input:,.2f} will be queued for auditor review."
            )

        justification = ""
        if needs_justification:
            justification = st.text_area(
                "📄 Business Justification (required)",
                placeholder="Client name, project code, attendees, pre-approval reference...",
                key="uc1_just",
            )

        can_submit = not (needs_justification and not justification.strip())
        if needs_justification and not justification.strip():
            st.caption("⬆️ Fill in the justification to enable submission.")

        st.markdown("")
        if st.button("🚀 Submit Expense", disabled=not can_submit,
                      type="primary", width="stretch", key="uc1_submit"):
            status = "Auto-Approved" if is_auto else "Pending Approval"
            new_exp = {
                "id": gen_id("EXP"), "employee": "You", "amount": amount_input,
                "date": date_input, "category": category, "description": description,
                "status": status, "justification": justification,
                "auditor_notes": "", "follow_up": None,
                "image_bytes": file_bytes, "mime_type": mime_type, "messages": [],
            }
            st.session_state.te_expenses.append(new_exp)
            if is_auto:
                st.success(f"🎉 **{new_exp['id']}** auto-approved! (${amount_input:,.2f})")
                st.balloons()
            else:
                st.success(f"📨 **{new_exp['id']}** submitted for review. Status: **{status}**")
            st.info("💡 Switch to **Reviewer** role to see this in the Pending Queue.")


# ──────────────────────────────────────────────────────────────
# UC1 — REVIEWER (Auditor)
# ──────────────────────────────────────────────────────────────
def uc1_reviewer():
    expenses = st.session_state.te_expenses
    policies = st.session_state.te_policies

    st.markdown(
        '<div class="hero">'
        "<h1>🔍 Expense Audit Dashboard</h1>"
        "<p>Review submissions, configure policies, and let AI flag anomalies.</p>"
        "</div>", unsafe_allow_html=True,
    )

    # ── KPI Row ──
    pending = [e for e in expenses if e["status"] == "Pending Approval"]
    approved = [e for e in expenses if e["status"] in ("Approved", "Auto-Approved")]
    total_amt = sum(e["amount"] for e in expenses)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(stat_card(str(len(pending)), "Pending Review"), unsafe_allow_html=True)
    with k2:
        st.markdown(stat_card(str(len(approved)), "Approved", "#16a34a"), unsafe_allow_html=True)
    with k3:
        st.markdown(stat_card(str(len(expenses)), "Total Expenses"), unsafe_allow_html=True)
    with k4:
        st.markdown(stat_card(f"${total_amt:,.0f}", "Total Spend", "#7c3aed"), unsafe_allow_html=True)
    st.markdown("")

    tab_queue, tab_settings = st.tabs(["📋 Pending Queue", "⚙️ Expense Policies Settings"])

    # ── Settings Tab ─────────────────────────────
    with tab_settings:
        st.markdown("### ⚙️ Configure Audit Policies")
        st.markdown(
            "> These thresholds are **injected into the Gemini prompt** so the AI "
            "auto-flags anomalies and employees are required to justify large expenses."
        )
        with st.form("te_policy_form"):
            ca, cb = st.columns(2)
            with ca:
                new_auto = st.number_input(
                    "💚 Auto-Approve Limit ($)", min_value=0.0, step=10.0,
                    value=policies["auto_approve_limit"],
                    help="Expenses at or below this amount are instantly approved.",
                )
            with cb:
                new_flag = st.number_input(
                    "🚩 Flag Threshold ($)", min_value=0.0, step=50.0,
                    value=policies["flag_threshold"],
                    help="Expenses above this amount require employee justification.",
                )
            st.markdown("")
            if st.form_submit_button("💾 Save Policies", type="primary", width="stretch"):
                st.session_state.te_policies["auto_approve_limit"] = new_auto
                st.session_state.te_policies["flag_threshold"] = new_flag
                st.success(f"✅ Updated — Auto-approve ≤ **${new_auto:.0f}**, Flag > **${new_flag:.0f}**")

        st.divider()
        st.markdown("#### 📖 How It Works")
        st.markdown("""
| Expense Amount | System Behavior |
|---|---|
| ≤ Auto-Approve Limit | ✅ Instantly approved |
| Between limits | 📋 Sent to Pending Queue for manual review |
| > Flag Threshold | 🚩 Employee must justify before submitting |
        """)

    # ── Queue Tab ────────────────────────────────
    with tab_queue:
        st.markdown("### 📋 All Expenses")
        df = pd.DataFrame([{
            "ID": e["id"], "Employee": e["employee"], "Amount": f"${e['amount']:,.2f}",
            "Date": e["date"], "Category": e["category"], "Status": e["status"],
        } for e in expenses])
        st.dataframe(style_status_col(df), width="stretch", hide_index=True)

        st.divider()

        reviewable = [e for e in expenses if e["status"] in ("Pending Approval", "Flagged - Justification Required")]
        if not reviewable:
            st.markdown(
                '<div class="empty-state">'
                '🎉 <strong>All caught up!</strong><br/>No pending expenses to review.</div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown("### 🔎 Review Expense")
        opts = {f"{e['id']} — {e['employee']} — ${e['amount']:,.2f}": e["id"] for e in reviewable}
        sel_label = st.selectbox("Select expense:", list(opts.keys()))
        sel_id = opts[sel_label]
        exp = next(e for e in expenses if e["id"] == sel_id)
        exp_idx = next(i for i, e in enumerate(expenses) if e["id"] == sel_id)

        st.markdown("")
        col_l, col_r = st.columns([1, 1.5])

        # ── Left: Receipt + Data ──
        with col_l:
            st.markdown("#### 🖼️ Receipt")
            if exp.get("image_bytes") and exp.get("mime_type", "").startswith("image"):
                st.image(exp["image_bytes"], width="stretch")
            elif exp.get("image_bytes"):
                st.info("📄 PDF document attached")
            else:
                st.markdown(
                    '<div class="placeholder-box">'
                    '🖼️<br/>No receipt image<br/>'
                    '<span style="font-size:0.75rem">Demo record</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")
            st.markdown("#### 📊 Extracted Data")
            st.markdown(
                f'<div class="info-block">'
                f'<strong>💰 Amount:</strong> ${exp["amount"]:,.2f}<br/>'
                f'<strong>📅 Date:</strong> {exp["date"]}<br/>'
                f'<strong>🏷️ Category:</strong> {exp["category"]}<br/>'
                f'<strong>📝 Description:</strong> {exp["description"]}</div>',
                unsafe_allow_html=True,
            )
            if exp.get("justification"):
                st.markdown("#### 📄 Employee Justification")
                st.info(exp["justification"])

        # ── Right: AI Analysis + Actions ──
        with col_r:
            st.markdown("#### 🤖 Gemini AI Analysis")

            # ── REAL Gemini call ──
            with st.spinner("🧠 Gemini is analyzing this expense..."):
                prompt = f"""You are a corporate expense auditor AI.

Expense: {exp['id']}
Employee: {exp['employee']}
Amount: ${exp['amount']:,.2f}
Date: {exp['date']}
Category: {exp['category']}
Description: {exp['description']}
Justification: {exp.get('justification') or 'None provided'}

Policy: Auto-approve ≤ ${policies['auto_approve_limit']:.0f}, Flag > ${policies['flag_threshold']:.0f}.

Assess reasonableness, policy compliance, and flag concerns.
Recommend: APPROVE, REJECT, or REQUEST MORE INFO. 4-5 sentences, professional."""
                try:
                    analysis = call_gemini(prompt)
                except Exception as e:
                    analysis = f"⚠️ Gemini unavailable: {e}"

            if "reject" in analysis.lower():
                st.error(f"🤖 **AI Assessment:**\n\n{analysis}")
            elif "request more" in analysis.lower() or "more info" in analysis.lower():
                st.warning(f"🤖 **AI Assessment:**\n\n{analysis}")
            else:
                st.info(f"🤖 **AI Assessment:**\n\n{analysis}")

            st.divider()
            st.markdown("#### ✍️ Reviewer Actions")

            at1, at2, at3 = st.tabs(["📌 Status", "💬 Communicate", "📝 Internal Notes"])

            with at1:
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button("✅ Approve", key=f"te_apr_{exp['id']}", width="stretch", type="primary"):
                        st.session_state.te_expenses[exp_idx]["status"] = "Approved"
                        st.balloons()
                        st.rerun()
                with bc2:
                    if st.button("❌ Reject", key=f"te_rej_{exp['id']}", width="stretch"):
                        st.session_state.te_expenses[exp_idx]["status"] = "Rejected"
                        st.rerun()
                with bc3:
                    if st.button("⏳ Keep Pending", key=f"te_pen_{exp['id']}", width="stretch"):
                        st.info("Kept as pending.")

            with at2:
                st.markdown(f"**Send message to:** {exp['employee']}")
                channel = st.radio("Channel", ["📧 Email", "💬 Chat"], horizontal=True, key=f"te_ch_{exp['id']}")
                msg = st.text_area(
                    "Message",
                    placeholder="e.g., Could you provide the project code for this expense?",
                    key=f"te_msg_{exp['id']}",
                )
                if st.button("📤 Send", key=f"te_send_{exp['id']}", width="stretch"):
                    if msg.strip():
                        st.session_state.te_expenses[exp_idx].setdefault("messages", []).append({
                            "channel": channel, "text": msg,
                            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "from": "Auditor",
                        })
                        st.success(f"✅ Sent via {channel}")
                    else:
                        st.warning("Enter a message first.")
                msgs = exp.get("messages", [])
                if msgs:
                    st.markdown("---")
                    st.markdown("**📜 Message History**")
                    for m in reversed(msgs):
                        st.caption(f"{m['sent_at']} — {m['channel']} — {m['from']}")
                        st.markdown(f"> {m['text']}")

            with at3:
                note = st.text_area(
                    "Notes", value=exp.get("auditor_notes", ""),
                    placeholder="e.g., Confirmed project code with employee.",
                    key=f"te_note_{exp['id']}",
                )
                followup = st.date_input(
                    "📅 Follow-up Reminder",
                    value=exp.get("follow_up") or date.today() + timedelta(days=3),
                    key=f"te_fu_{exp['id']}",
                )
                if st.button("💾 Save Notes", key=f"te_sn_{exp['id']}", width="stretch"):
                    st.session_state.te_expenses[exp_idx]["auditor_notes"] = note
                    st.session_state.te_expenses[exp_idx]["follow_up"] = followup
                    st.success(f"✅ Notes saved. Follow-up: **{followup}**")


# ╔══════════════════════════════════════════════════════════════╗
# ║                                                              ║
# ║  USE CASE 2 — Accounts Payable  (Vendor Invoices)           ║
# ║                                                              ║
# ╚══════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────
# UC2 — SUBMITTER (Vendor / Procurement)
# ──────────────────────────────────────────────────────────────
def uc2_submitter():
    invoices = st.session_state.ap_invoices
    po_db = st.session_state.ap_po_database

    st.markdown(
        '<div class="hero-ap">'
        "<h1>📑 Vendor Invoice Portal</h1>"
        "<p>Upload B2B invoices, link to Purchase Orders, and track payment status.</p>"
        "</div>", unsafe_allow_html=True,
    )

    tab_upload, tab_status = st.tabs(["📤 Upload B2B Invoice", "📜 Invoice Status"])

    # ── Status Tab ───────────────────────────────
    with tab_status:
        st.markdown("### 📜 Invoice Status Tracker")
        if not invoices:
            st.info("No invoices submitted yet.")
        else:
            df = pd.DataFrame([{
                "ID": inv["id"], "Vendor": inv["vendor"], "Invoice #": inv["invoice_number"],
                "Amount": f"${inv['amount']:,.2f}", "PO #": inv.get("po_number", "—"),
                "Due Date": inv["due_date"], "Status": inv["status"],
            } for inv in invoices])
            st.dataframe(style_status_col(df), width="stretch", hide_index=True)

    # ── Upload Tab ───────────────────────────────
    with tab_upload:
        st.markdown("### 📤 Upload Vendor Invoice")
        st.markdown("Link invoice to a PO and let Document AI extract line items automatically.")

        uc1, uc2_col = st.columns([1, 1])
        with uc1:
            po_input = st.selectbox(
                "📋 Purchase Order Number",
                ["(None — no PO)"] + list(po_db.keys()),
                key="ap_po_select",
            )
        with uc2_col:
            ap_terms = st.selectbox("📋 Payment Terms", ["Net 30", "Net 45", "Net 60", "Due on Receipt"], key="ap_terms")

        ap_uploaded = st.file_uploader(
            "Drop vendor invoice here (PNG, JPG, PDF)",
            type=["png", "jpg", "jpeg", "pdf", "tif", "tiff"],
            key="uc2_upload",
        )

        if ap_uploaded is None:
            return

        file_bytes = ap_uploaded.read()
        mime = get_mime(ap_uploaded.name)

        col_img, col_data = st.columns([1, 1.5])
        with col_img:
            if mime.startswith("image"):
                st.image(file_bytes, width="stretch", caption="Uploaded Invoice")
            else:
                st.info("📄 PDF uploaded")

        with col_data:
            # ── REAL Document AI call ──
            with st.spinner("🔍 Document AI extracting invoice data..."):
                try:
                    extracted = extract_with_docai(file_bytes, mime)
                except Exception as e:
                    st.error(f"Document AI error: {e}")
                    extracted = {}

            st.markdown("##### ✏️ Verify Extracted Data")
            ac1, ac2 = st.columns(2)
            with ac1:
                v_name = st.text_input("🏢 Vendor",
                    value=extracted.get("supplier_name", ""), key="ap_vendor")
            with ac2:
                v_inv = st.text_input("📄 Invoice #",
                    value=extracted.get("invoice_id", ""), key="ap_invnum")

            ac3, ac4 = st.columns(2)
            with ac3:
                v_amt = st.number_input("💰 Total Amount ($)",
                    value=parse_amount(extracted.get("total_amount", "")),
                    min_value=0.0, step=1.0, key="ap_amt")
            with ac4:
                v_cur = st.selectbox("💱 Currency", ["USD", "TWD", "EUR", "GBP", "JPY"], key="ap_cur")

            ac5, ac6 = st.columns(2)
            with ac5:
                v_date = st.text_input("📅 Invoice Date",
                    value=extracted.get("invoice_date", str(date.today())), key="ap_date")
            with ac6:
                v_due = st.text_input("⏰ Due Date",
                    value=extracted.get("due_date", ""), key="ap_due")

            v_cat = st.selectbox("🏷️ Category",
                ["IT Hardware", "Cloud Services", "Office Supplies",
                 "Professional Services", "Marketing", "Facilities", "Other"],
                key="ap_cat")

        # ── Line-Item Extraction (from Document AI nested entities) ──
        st.divider()
        st.markdown("##### 📦 Extracted Line Items")
        st.caption("Document AI extracts table data from invoices. Verify and edit below.")

        # Build DataFrame from Document AI's _line_items
        raw_lines = extracted.get("_line_items", [])
        if raw_lines:
            li_rows = []
            for li in raw_lines:
                li_rows.append({
                    "Description": li.get("description", ""),
                    "Qty": int(parse_amount(li.get("quantity", "1"))),
                    "Unit Price": parse_amount(li.get("unit_price", "0")),
                    "Total": parse_amount(li.get("amount", "0")),
                })
            default_li_df = pd.DataFrame(li_rows)
        else:
            default_li_df = pd.DataFrame([
                {"Description": "", "Qty": 1, "Unit Price": 0.00, "Total": 0.00},
            ])

        edited_li = st.data_editor(
            default_li_df,
            num_rows="dynamic",
            width=800,
            key="ap_li_editor",
        )

        # ── PO Preview ──
        po_num = po_input if po_input != "(None — no PO)" else ""
        if po_num and po_num in po_db:
            st.divider()
            po = po_db[po_num]
            st.markdown(f"##### 🔗 Linked PO: **{po_num}** — {po['vendor']}")
            st.markdown(f"PO Total: **${po['total']:,.2f}** | Status: **{po['status']}**")
            po_li_df = pd.DataFrame(po["line_items"])
            st.dataframe(po_li_df, width="stretch", hide_index=True)

        st.markdown("")
        if st.button("🚀 Submit Invoice", type="primary", width="stretch", key="uc2_submit"):
            # Build line_items list from editor
            li_list = []
            for _, row in edited_li.iterrows():
                li_list.append({
                    "description": str(row.get("Description", "")),
                    "qty": int(row.get("Qty", 1)),
                    "unit_price": float(row.get("Unit Price", 0)),
                    "total": float(row.get("Total", 0)),
                })

            new_inv = {
                "id": gen_id("INV"), "vendor": v_name or "Unknown Vendor",
                "invoice_number": v_inv or "N/A", "amount": v_amt,
                "currency": v_cur, "date": v_date,
                "due_date": v_due or "N/A", "po_number": po_num,
                "category": v_cat, "payment_terms": ap_terms,
                "status": "Pending Review", "line_items": li_list,
                "notes": "", "image_bytes": file_bytes, "mime_type": mime,
            }
            st.session_state.ap_invoices.append(new_inv)
            st.success(f"📨 Invoice **{new_inv['id']}** from **{v_name}** submitted! (${v_amt:,.2f})")
            st.info("💡 Switch to **Reviewer** role to process this invoice.")


# ──────────────────────────────────────────────────────────────
# UC2 — REVIEWER (AP Specialist)
# ──────────────────────────────────────────────────────────────
def uc2_reviewer():
    invoices = st.session_state.ap_invoices
    po_db = st.session_state.ap_po_database
    settings = st.session_state.ap_settings
    tolerance = settings["price_tolerance_pct"]

    st.markdown(
        '<div class="hero-ap">'
        "<h1>📑 AP Review Dashboard</h1>"
        "<p>3-way match verification, AI anomaly detection, and vendor communication.</p>"
        "</div>", unsafe_allow_html=True,
    )

    # ── KPI Row ──
    pend = [i for i in invoices if i["status"] == "Pending Review"]
    flagged = [i for i in invoices if "Flagged" in i["status"]]
    approved_pay = [i for i in invoices if "Approved" in i["status"]]
    total_payable = sum(i["amount"] for i in invoices if i["status"] != "Rejected")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(stat_card(str(len(pend)), "Pending Review"), unsafe_allow_html=True)
    with k2:
        st.markdown(stat_card(str(len(flagged)), "Flagged", "#f59e0b"), unsafe_allow_html=True)
    with k3:
        st.markdown(stat_card(str(len(approved_pay)), "Approved", "#16a34a"), unsafe_allow_html=True)
    with k4:
        st.markdown(stat_card(f"${total_payable:,.0f}", "Total Payable", "#0d9488"), unsafe_allow_html=True)
    st.markdown("")

    tab_inbox, tab_settings = st.tabs(["📋 AP Inbox", "⚙️ PO Database & Tolerance Settings"])

    # ── Settings Tab ─────────────────────────────
    with tab_settings:
        st.markdown("### ⚙️ PO Database & Tolerance Settings")

        st.markdown("#### 📋 Purchase Order Database")
        po_rows = []
        for po_id, po in po_db.items():
            po_rows.append({
                "PO #": po_id, "Vendor": po["vendor"],
                "Description": po["description"],
                "Total": f"${po['total']:,.2f}", "Status": po["status"],
            })
        st.dataframe(style_status_col(pd.DataFrame(po_rows)), width="stretch", hide_index=True)

        st.divider()
        st.markdown("#### 🎚️ Price Tolerance Threshold")
        st.markdown(
            "> If an invoice amount exceeds the linked PO total by more than this percentage, "
            "it will be **automatically flagged** for review."
        )
        new_tol = st.slider(
            "Price Tolerance (%)", min_value=0.0, max_value=25.0, step=0.5,
            value=tolerance, key="ap_tol_slider",
        )
        if new_tol != tolerance:
            st.session_state.ap_settings["price_tolerance_pct"] = new_tol
            st.success(f"✅ Tolerance updated to **{new_tol:.1f}%**")

        st.divider()
        st.markdown("#### 📖 How 3-Way Matching Works")
        st.markdown("""
| Check | What it verifies |
|---|---|
| **PO Match** | Invoice references a valid, open Purchase Order |
| **Line-Item Match** | Invoice line items align with PO line items (qty, description) |
| **Price Match** | Invoice total is within tolerance % of the PO total |
        """)

    # ── Inbox Tab ────────────────────────────────
    with tab_inbox:
        st.markdown("### 📋 All Vendor Invoices")
        inv_df = pd.DataFrame([{
            "ID": i["id"], "Vendor": i["vendor"], "Invoice #": i["invoice_number"],
            "Amount": f"${i['amount']:,.2f}", "PO #": i.get("po_number") or "—",
            "Due": i["due_date"], "Status": i["status"],
        } for i in invoices])
        st.dataframe(style_status_col(inv_df), width="stretch", hide_index=True)

        st.divider()

        reviewable = [i for i in invoices if i["status"] in ("Pending Review", "Flagged - Amount Mismatch")]
        if not reviewable:
            st.markdown(
                '<div class="empty-state">'
                '🎉 <strong>All caught up!</strong><br/>No invoices pending review.</div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown("### 🔎 Review Invoice")
        opts = {f"{i['id']} — {i['vendor']} — ${i['amount']:,.2f}": i["id"] for i in reviewable}
        sel = st.selectbox("Select invoice:", list(opts.keys()), key="ap_inv_sel")
        inv = next(i for i in invoices if i["id"] == opts[sel])
        inv_idx = next(idx for idx, i in enumerate(invoices) if i["id"] == opts[sel])

        st.markdown("")
        col_l, col_r = st.columns([1, 1.5])

        # ── Left: Invoice Details + Line Items ──
        with col_l:
            st.markdown("#### 🖼️ Invoice Document")
            if inv.get("image_bytes") and inv.get("mime_type", "").startswith("image"):
                st.image(inv["image_bytes"], width="stretch")
            elif inv.get("image_bytes"):
                st.info("📄 PDF invoice attached")
            else:
                st.markdown(
                    '<div class="placeholder-box">'
                    '📄<br/>No document preview<br/>'
                    '<span style="font-size:0.75rem">Demo record</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.markdown("#### 📊 Invoice Data")
            st.markdown(
                f'<div class="info-block-ap">'
                f'<strong>🏢 Vendor:</strong> {inv["vendor"]}<br/>'
                f'<strong>📄 Invoice #:</strong> {inv["invoice_number"]}<br/>'
                f'<strong>💰 Amount:</strong> ${inv["amount"]:,.2f} {inv["currency"]}<br/>'
                f'<strong>📅 Date:</strong> {inv["date"]}<br/>'
                f'<strong>⏰ Due:</strong> {inv["due_date"]}<br/>'
                f'<strong>📋 Terms:</strong> {inv["payment_terms"]}<br/>'
                f'<strong>🏷️ Category:</strong> {inv["category"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Extracted Line Items ──
            if inv.get("line_items"):
                st.markdown("#### 📦 Invoice Line Items")
                li_df = pd.DataFrame(inv["line_items"])
                st.dataframe(li_df, width="stretch", hide_index=True)

            # ── PO Comparison ──
            po_num = inv.get("po_number", "")
            if po_num and po_num in po_db:
                po = po_db[po_num]
                st.markdown(f"#### 🔗 PO: {po_num}")
                st.markdown(f"**Vendor:** {po['vendor']} | **PO Total:** ${po['total']:,.2f}")
                po_df = pd.DataFrame(po["line_items"])
                st.dataframe(po_df, width="stretch", hide_index=True)

                # Variance
                variance = inv["amount"] - po["total"]
                var_pct = (variance / po["total"] * 100) if po["total"] else 0
                if abs(var_pct) <= tolerance:
                    st.success(f"✅ Within tolerance — Variance: ${variance:,.2f} ({var_pct:+.1f}%)")
                else:
                    st.error(
                        f"🚨 **OVER TOLERANCE** — Invoice ${inv['amount']:,.2f} vs PO ${po['total']:,.2f}\n\n"
                        f"Variance: **${variance:,.2f}** ({var_pct:+.1f}%) exceeds {tolerance:.1f}% threshold"
                    )
            elif po_num:
                st.warning(f"⚠️ PO **{po_num}** not found in system")
            else:
                st.warning("⚠️ No PO linked to this invoice")

        # ── Right: AI Analysis + Actions ──
        with col_r:
            st.markdown("#### 🤖 Gemini AI Invoice Analysis")

            # Build PO context for prompt
            po_context = "No PO linked."
            if po_num and po_num in po_db:
                po = po_db[po_num]
                po_items = "\n".join([f"  - {li['item']}: {li['qty']} x ${li['unit_price']:,.2f}" for li in po["line_items"]])
                po_context = f"PO {po_num}: Vendor={po['vendor']}, Total=${po['total']:,.2f}\nPO Line Items:\n{po_items}"

            inv_items = "\n".join([
                f"  - {li['description']}: {li['qty']} x ${li['unit_price']:,.2f} = ${li['total']:,.2f}"
                for li in inv.get("line_items", [])
            ])

            # ── REAL Gemini call ──
            with st.spinner("🧠 Gemini is analyzing this invoice..."):
                ap_prompt = f"""You are an Accounts Payable AI analyst for a corporate finance team.

INVOICE:
- ID: {inv['id']}
- Vendor: {inv['vendor']}
- Invoice #: {inv['invoice_number']}
- Total: ${inv['amount']:,.2f} {inv['currency']}
- Date: {inv['date']} | Due: {inv['due_date']} | Terms: {inv['payment_terms']}
- Category: {inv['category']}
- Invoice Line Items:
{inv_items}

PURCHASE ORDER:
{po_context}

COMPANY POLICY:
- Price tolerance threshold: {tolerance:.1f}% (invoice can exceed PO by up to this %)

INSTRUCTIONS:
1. Perform 3-way match: PO exists? Line items match? Amounts within tolerance?
2. Flag specific line-item discrepancies (e.g., price increases, unexpected charges).
3. Check for duplicate risk, missing details, or unusual patterns.
4. If there IS a discrepancy, state the EXACT numbers (e.g., "billed $X vs PO approved $Y").
5. Recommend: APPROVE FOR PAYMENT, HOLD FOR REVIEW, or REJECT.
6. 5-6 sentences, professional tone."""

                try:
                    ap_analysis = call_gemini(ap_prompt)
                except Exception as e:
                    ap_analysis = f"⚠️ Gemini unavailable: {e}"

            if "reject" in ap_analysis.lower():
                st.error(f"🤖 **AI Assessment:**\n\n{ap_analysis}")
            elif "hold" in ap_analysis.lower():
                st.warning(f"🤖 **AI Assessment:**\n\n{ap_analysis}")
            else:
                st.info(f"🤖 **AI Assessment:**\n\n{ap_analysis}")

            st.divider()
            st.markdown("#### ✍️ AP Actions")

            act1, act2, act3 = st.tabs(["📌 Status", "📝 Notes", "📧 Vendor Comms"])

            # ── Status ──
            with act1:
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("✅ Approve for Payment", key=f"ap_apr_{inv['id']}", width="stretch", type="primary"):
                        st.session_state.ap_invoices[inv_idx]["status"] = "Approved for Payment"
                        st.balloons()
                        st.rerun()
                with b2:
                    if st.button("🔄 Hold", key=f"ap_hold_{inv['id']}", width="stretch"):
                        st.session_state.ap_invoices[inv_idx]["status"] = "On Hold"
                        st.rerun()
                with b3:
                    if st.button("❌ Reject", key=f"ap_rej_{inv['id']}", width="stretch"):
                        st.session_state.ap_invoices[inv_idx]["status"] = "Rejected"
                        st.rerun()

            # ── Notes ──
            with act2:
                ap_note = st.text_area(
                    "Internal Notes", value=inv.get("notes", ""),
                    placeholder="e.g., Confirmed overage with procurement.",
                    key=f"ap_note_{inv['id']}",
                )
                if st.button("💾 Save Notes", key=f"ap_sn_{inv['id']}", width="stretch"):
                    st.session_state.ap_invoices[inv_idx]["notes"] = ap_note
                    st.success("✅ Notes saved.")

            # ── Vendor Comms ──
            with act3:
                st.markdown(f"**Draft email to:** {inv['vendor']}")

                # Pre-draft an AI email if there's a discrepancy
                if po_num and po_num in po_db:
                    po = po_db[po_num]
                    variance = inv["amount"] - po["total"]
                    if variance > 0:
                        draft = (
                            f"Subject: Invoice {inv['invoice_number']} — Amount Discrepancy\n\n"
                            f"Dear {inv['vendor']} Accounts Receivable Team,\n\n"
                            f"We have received your invoice {inv['invoice_number']} dated {inv['date']} "
                            f"for ${inv['amount']:,.2f}. However, our Purchase Order {po_num} was "
                            f"approved for ${po['total']:,.2f}.\n\n"
                            f"This represents a variance of ${variance:,.2f} "
                            f"({variance/po['total']*100:.1f}% over the PO amount).\n\n"
                            f"Could you please provide an itemized breakdown of the additional charges "
                            f"so we can process this for approval?\n\n"
                            f"Best regards,\n"
                            f"AP Department"
                        )
                    else:
                        draft = f"Subject: Invoice {inv['invoice_number']} — Confirmation\n\nNo discrepancy found."
                else:
                    draft = (
                        f"Subject: Invoice {inv['invoice_number']} — PO Required\n\n"
                        f"Dear {inv['vendor']},\n\n"
                        f"We received your invoice but no valid PO is linked. "
                        f"Please provide the PO number for processing.\n\n"
                        f"Best regards,\nAP Department"
                    )

                email_body = st.text_area(
                    "Email Draft (AI pre-drafted)", value=draft,
                    height=250, key=f"ap_email_{inv['id']}",
                )
                if st.button("📤 Send Email", key=f"ap_send_{inv['id']}", width="stretch"):
                    st.success(f"✅ Email sent to {inv['vendor']} (simulated)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ROUTER — dispatch to the correct view                      ║
# ╚══════════════════════════════════════════════════════════════╝
if is_uc1 and is_submitter:
    uc1_submitter()
elif is_uc1 and not is_submitter:
    uc1_reviewer()
elif not is_uc1 and is_submitter:
    uc2_submitter()
else:
    uc2_reviewer()
