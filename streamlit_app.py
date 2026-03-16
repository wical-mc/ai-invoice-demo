import os
import json
import uuid
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd

# ══════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Expense Processing System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #6d28d9 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }
    .hero::after {
        content: ''; position: absolute; top: -40%; right: -10%; width: 300px; height: 300px;
        background: rgba(255,255,255,0.06); border-radius: 50%;
    }
    .hero h1 { margin: 0 0 0.3rem; font-size: 1.5rem; position: relative; z-index: 1; }
    .hero p  { margin: 0; opacity: 0.85; font-size: 0.9rem; position: relative; z-index: 1; }

    .stat-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem 1.25rem; text-align: center;
    }
    .stat-card .num  { font-size: 1.6rem; font-weight: 700; color: #4f46e5; margin: 0; }
    .stat-card .lbl  { font-size: 0.78rem; color: #64748b; margin: 0.15rem 0 0; }

    .badge {
        display: inline-block; padding: 0.2rem 0.65rem; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .b-pending  { background: #fef3c7; color: #92400e; }
    .b-approved { background: #d1fae5; color: #065f46; }
    .b-rejected { background: #fee2e2; color: #991b1b; }
    .b-auto     { background: #dbeafe; color: #1e40af; }
    .b-flagged  { background: #fce7f3; color: #9d174d; }

    .info-block {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    .expense-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0.75rem; border-bottom: 1px solid #f1f5f9;
        border-radius: 8px; margin-bottom: 0.25rem;
    }
    .expense-row:hover { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GCP Clients (lazy-loaded, cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_docai_client():
    """Initialize Document AI client with regional endpoint."""
    from google.cloud import documentai_v1 as documentai
    project_id = os.environ.get("PROJECT_ID", "mcps-owntest")
    location = os.environ.get("LOCATION", "us")
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    return client, project_id, location


@st.cache_resource
def get_gemini_model():
    """Initialize Vertex AI Gemini model."""
    import vertexai
    from vertexai.generative_models import GenerativeModel
    project_id = os.environ.get("PROJECT_ID", "mcps-owntest")
    vertexai.init(project=project_id, location="us-central1")
    return GenerativeModel("gemini-2.5-flash")


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def extract_with_docai(file_bytes: bytes, mime_type: str) -> dict:
    """Call Document AI to extract entities from a document."""
    from google.cloud import documentai_v1 as documentai
    client, project_id, location = get_docai_client()
    processor_id = os.environ.get("PROCESSOR_ID", "ca7b526548cdf4ef")
    resource_name = client.processor_path(project_id, location, processor_id)
    request = documentai.ProcessRequest(
        name=resource_name,
        raw_document=documentai.RawDocument(content=file_bytes, mime_type=mime_type),
    )
    result = client.process_document(request=request)
    extracted = {}
    for entity in result.document.entities:
        extracted[entity.type_] = entity.mention_text
    extracted["_full_text"] = (result.document.text or "")[:2000]
    return extracted


def call_gemini(prompt: str) -> str:
    """Send prompt to Gemini and return the text response."""
    model = get_gemini_model()
    return model.generate_content(prompt).text


def get_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {"pdf": "application/pdf", "png": "image/png",
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "tif": "image/tiff", "tiff": "image/tiff"}.get(ext, "application/pdf")


def badge_html(status: str) -> str:
    css = {"Pending Approval": "b-pending", "Approved": "b-approved",
           "Rejected": "b-rejected", "Auto-Approved": "b-auto",
           "Flagged - Justification Required": "b-flagged"}.get(status, "b-pending")
    return f'<span class="badge {css}">{status}</span>'


def gen_id() -> str:
    return f"EXP-{uuid.uuid4().hex[:6].upper()}"


# ══════════════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════════════
if "policies" not in st.session_state:
    st.session_state.policies = {
        "auto_approve_limit": 50.0,
        "flag_threshold": 200.0,
    }

if "ap_invoices" not in st.session_state:
    st.session_state.ap_invoices = [
        {
            "id": "INV-001",
            "vendor": "TechSupply Co.",
            "invoice_number": "TS-2026-0412",
            "amount": 12500.00,
            "currency": "USD",
            "date": "2026-03-05",
            "due_date": "2026-04-04",
            "po_number": "PO-2026-088",
            "po_match": True,
            "po_amount": 12500.00,
            "category": "IT Hardware",
            "payment_terms": "Net 30",
            "status": "Approved for Payment",
            "line_items": [
                {"description": "Dell Monitor 27\" 4K", "qty": 5, "unit_price": 2500.00},
            ],
            "notes": "Matched to PO. Goods received confirmed.",
            "image_bytes": None,
            "mime_type": None,
        },
        {
            "id": "INV-002",
            "vendor": "CloudServ Inc.",
            "invoice_number": "CS-88721",
            "amount": 4200.00,
            "currency": "USD",
            "date": "2026-03-08",
            "due_date": "2026-04-07",
            "po_number": "PO-2026-095",
            "po_match": True,
            "po_amount": 4000.00,
            "category": "Cloud Services",
            "payment_terms": "Net 30",
            "status": "Pending Review",
            "line_items": [
                {"description": "Monthly cloud hosting (March)", "qty": 1, "unit_price": 3800.00},
                {"description": "Overage charges", "qty": 1, "unit_price": 400.00},
            ],
            "notes": "",
            "image_bytes": None,
            "mime_type": None,
        },
        {
            "id": "INV-003",
            "vendor": "Office Essentials Ltd.",
            "invoice_number": "OE-2026-3371",
            "amount": 875.50,
            "currency": "USD",
            "date": "2026-03-10",
            "due_date": "2026-04-09",
            "po_number": "",
            "po_match": False,
            "po_amount": 0.0,
            "category": "Office Supplies",
            "payment_terms": "Net 30",
            "status": "Pending Review",
            "line_items": [
                {"description": "A4 Copy Paper (50 reams)", "qty": 50, "unit_price": 12.50},
                {"description": "Toner Cartridge HP 26A", "qty": 3, "unit_price": 41.83},
            ],
            "notes": "",
            "image_bytes": None,
            "mime_type": None,
        },
        {
            "id": "INV-004",
            "vendor": "Acme Consulting",
            "invoice_number": "AC-9002",
            "amount": 28000.00,
            "currency": "USD",
            "date": "2026-03-01",
            "due_date": "2026-03-31",
            "po_number": "PO-2026-072",
            "po_match": True,
            "po_amount": 25000.00,
            "category": "Professional Services",
            "payment_terms": "Net 30",
            "status": "Flagged - Amount Mismatch",
            "line_items": [
                {"description": "Strategy consulting (Feb)", "qty": 1, "unit_price": 25000.00},
                {"description": "Additional research hours", "qty": 1, "unit_price": 3000.00},
            ],
            "notes": "Invoice amount exceeds PO by $3,000. Needs manager approval.",
            "image_bytes": None,
            "mime_type": None,
        },
    ]

if "expenses" not in st.session_state:
    st.session_state.expenses = [
        {
            "id": "EXP-A1B2C3",
            "employee_name": "Alice Chen",
            "amount": 42.50,
            "date": "2026-03-10",
            "category": "Meals & Entertainment",
            "description": "Team lunch — quarterly planning",
            "status": "Auto-Approved",
            "justification": "",
            "auditor_notes": "",
            "follow_up_date": None,
            "image_bytes": None,
            "mime_type": None,
            "messages": [],
        },
        {
            "id": "EXP-D4E5F6",
            "employee_name": "Bob Wang",
            "amount": 350.00,
            "date": "2026-03-11",
            "category": "Travel & Transportation",
            "description": "Client visit — Taipei round-trip HSR",
            "status": "Pending Approval",
            "justification": "Client onsite meeting for Project Falcon. Pre-approved by PM.",
            "auditor_notes": "",
            "follow_up_date": None,
            "image_bytes": None,
            "mime_type": None,
            "messages": [],
        },
        {
            "id": "EXP-G7H8I9",
            "employee_name": "Carol Liu",
            "amount": 1280.00,
            "date": "2026-03-08",
            "category": "Equipment & Supplies",
            "description": "Ergonomic keyboard + monitor stand",
            "status": "Approved",
            "justification": "RSI prevention — doctor's recommendation attached.",
            "auditor_notes": "Verified medical note. Within annual IT budget.",
            "follow_up_date": None,
            "image_bytes": None,
            "mime_type": None,
            "messages": [],
        },
    ]


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧾 AI Invoice & Expense System")
    st.divider()

    st.markdown("#### 📂 Use Case")
    use_case = st.radio(
        "use_case",
        ["UC1: T&E Employee Expenses", "UC2: Accounts Payable"],
        label_visibility="collapsed",
    )

    st.divider()

    if "UC1" in use_case:
        st.markdown("#### 👤 Select Role")
        role = st.radio(
            "role", ["Employee (Submitter)", "Auditor (Reviewer)"],
            label_visibility="collapsed",
        )

        st.divider()

        # Live policy summary
        p = st.session_state.policies
        st.markdown("#### ⚙️ Active Policies")
        st.caption(f"Auto-approve: **≤ ${p['auto_approve_limit']:.0f}**")
        st.caption(f"Flag threshold: **> ${p['flag_threshold']:.0f}**")
    else:
        role = None
        st.markdown("#### 👤 Role")
        ap_role = st.radio(
            "ap_role", ["AP Clerk", "AP Manager"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("#### 📊 AP Summary")
        ap_invs = st.session_state.get("ap_invoices", [])
        st.caption(f"Total invoices: **{len(ap_invs)}**")
        pending_ap = len([i for i in ap_invs if i["status"] == "Pending Review"])
        st.caption(f"Pending review: **{pending_ap}**")

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:0.72rem;'>"
        "Powered by<br/>Document AI · Gemini 2.5 Flash · Streamlit"
        "</div>", unsafe_allow_html=True,
    )

is_auditor = role is not None and "Auditor" in role
policies = st.session_state.policies
expenses = st.session_state.expenses


# ══════════════════════════════════════════════════════════════
# USE CASE 1: T&E Employee Expenses
# ══════════════════════════════════════════════════════════════
if "UC1" not in use_case:
    pass  # Skip UC1 — handled below for UC2
elif is_auditor:

    st.markdown(
        '<div class="hero">'
        "<h1>🔍 Expense Audit Dashboard</h1>"
        "<p>Review submissions, configure policies, and let AI flag anomalies automatically.</p>"
        "</div>", unsafe_allow_html=True,
    )

    # ── KPI Row ──
    pending = [e for e in expenses if e["status"] == "Pending Approval"]
    approved = [e for e in expenses if e["status"] in ("Approved", "Auto-Approved")]
    flagged = [e for e in expenses if "Flag" in e["status"]]
    total_amount = sum(e["amount"] for e in expenses)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="stat-card"><p class="num">{len(pending)}</p><p class="lbl">Pending Review</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-card"><p class="num">{len(approved)}</p><p class="lbl">Approved</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-card"><p class="num">{len(flagged)}</p><p class="lbl">Flagged</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-card"><p class="num">${total_amount:,.0f}</p><p class="lbl">Total Spend</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──
    tab_queue, tab_policies = st.tabs(["📋 Pending Queue", "⚙️ Expense Policies Settings"])

    # ────────────────────────────────────────────
    # Tab: Policies
    # ────────────────────────────────────────────
    with tab_policies:
        st.markdown("### ⚙️ Configure Audit Policies")
        st.markdown(
            "> These thresholds are **injected into the Gemini prompt** to auto-flag anomalies "
            "and determine if employees need to provide extra justification during upload."
        )
        st.markdown("")

        with st.form("policy_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                new_auto = st.number_input(
                    "💚 Auto-Approve Limit ($)",
                    min_value=0.0, step=10.0,
                    value=policies["auto_approve_limit"],
                    help="Expenses at or below this amount are automatically approved — no reviewer action needed.",
                )
            with col_b:
                new_flag = st.number_input(
                    "🚩 Flag for Justification Threshold ($)",
                    min_value=0.0, step=50.0,
                    value=policies["flag_threshold"],
                    help="Expenses above this amount require the employee to provide a business justification before submission.",
                )

            st.markdown("")
            submitted = st.form_submit_button("💾 Save Policies", type="primary", width="stretch")
            if submitted:
                st.session_state.policies["auto_approve_limit"] = new_auto
                st.session_state.policies["flag_threshold"] = new_flag
                st.success(f"✅ Policies updated — Auto-approve ≤ **${new_auto:.0f}**, Flag > **${new_flag:.0f}**")

        st.divider()
        st.markdown("#### 📖 How It Works")
        st.markdown("""
| Expense Amount | System Behavior |
|---|---|
| ≤ Auto-Approve Limit | ✅ Instantly approved, no reviewer action |
| Between limits | 📋 Sent to Pending Queue for manual review |
| > Flag Threshold | 🚩 Employee must provide justification before submitting |
        """)

    # ────────────────────────────────────────────
    # Tab: Pending Queue
    # ────────────────────────────────────────────
    with tab_queue:
        st.markdown("### 📋 All Expenses")

        # Summary table
        df = pd.DataFrame([
            {
                "ID": e["id"],
                "Employee": e["employee_name"],
                "Amount": f"${e['amount']:,.2f}",
                "Date": e["date"],
                "Category": e.get("category", "—"),
                "Status": e["status"],
            }
            for e in expenses
        ])
        st.dataframe(df, width="stretch", hide_index=True)

        st.divider()

        # Select expense to review
        reviewable = [e for e in expenses if e["status"] in ("Pending Approval", "Flagged - Justification Required")]
        if not reviewable:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#94a3b8;">'
                "🎉 <strong>All caught up!</strong> No pending expenses to review."
                "</div>", unsafe_allow_html=True,
            )
        else:
            st.markdown("### 🔎 Review Expense")
            options = {f"{e['id']} — {e['employee_name']} — ${e['amount']:,.2f}": e["id"] for e in reviewable}
            selected_label = st.selectbox("Select an expense to review:", list(options.keys()))
            selected_id = options[selected_label]
            exp = next(e for e in expenses if e["id"] == selected_id)
            exp_idx = next(i for i, e in enumerate(expenses) if e["id"] == selected_id)

            st.markdown("")
            col_left, col_right = st.columns([1, 1.5])

            # ── Left: Receipt ──
            with col_left:
                st.markdown("#### 🖼️ Receipt")
                if exp.get("image_bytes"):
                    if exp["mime_type"] and exp["mime_type"].startswith("image"):
                        st.image(exp["image_bytes"], width="stretch")
                    else:
                        st.info("📄 PDF document attached")
                else:
                    st.markdown(
                        '<div style="background:#f1f5f9;border-radius:12px;padding:3rem;text-align:center;color:#94a3b8;">'
                        "🖼️<br/>Receipt image not available<br/><span style='font-size:0.75rem;'>Demo record — no file attached</span>"
                        "</div>", unsafe_allow_html=True,
                    )

                st.markdown("")
                st.markdown("#### 📊 Extracted Data")
                st.markdown(f"""
<div class="info-block">
<strong>💰 Amount:</strong> ${exp['amount']:,.2f}<br/>
<strong>📅 Date:</strong> {exp['date']}<br/>
<strong>🏷️ Category:</strong> {exp.get('category', 'N/A')}<br/>
<strong>📝 Description:</strong> {exp.get('description', 'N/A')}
</div>
                """, unsafe_allow_html=True)

                if exp.get("justification"):
                    st.markdown("#### 📄 Employee Justification")
                    st.info(exp["justification"])

            # ── Right: AI Analysis + Actions ──
            with col_right:
                st.markdown("#### 🤖 Gemini AI Analysis")

                # ── Real Gemini call using policy thresholds ──
                with st.spinner("🧠 Gemini is analyzing this expense..."):
                    gemini_prompt = f"""You are a corporate expense auditor AI assistant.

Expense Details:
- ID: {exp['id']}
- Employee: {exp['employee_name']}
- Amount: ${exp['amount']:,.2f}
- Date: {exp['date']}
- Category: {exp.get('category', 'N/A')}
- Description: {exp.get('description', 'N/A')}
- Employee Justification: {exp.get('justification', 'None provided')}

Company Policy Thresholds:
- Auto-approve limit: ${policies['auto_approve_limit']:.0f}
- Flag threshold (requires justification): ${policies['flag_threshold']:.0f}

Instructions:
1. Assess if this expense is reasonable given the category and description.
2. Check if the amount exceeds policy thresholds and if adequate justification was provided.
3. Flag any concerns (missing details, unusual patterns, policy violations).
4. Give a clear recommendation: APPROVE, REJECT, or REQUEST MORE INFO.
5. Keep your analysis to 4-5 sentences. Be professional but concise.
"""
                    try:
                        ai_analysis = call_gemini(gemini_prompt)
                    except Exception as e:
                        ai_analysis = f"⚠️ Gemini unavailable: {e}"

                if "reject" in ai_analysis.lower():
                    st.error(f"🤖 **AI Assessment:**\n\n{ai_analysis}")
                elif "request more" in ai_analysis.lower() or "more info" in ai_analysis.lower():
                    st.warning(f"🤖 **AI Assessment:**\n\n{ai_analysis}")
                else:
                    st.info(f"🤖 **AI Assessment:**\n\n{ai_analysis}")

                st.divider()
                st.markdown("#### ✍️ Reviewer Actions")

                action_tab1, action_tab2, action_tab3 = st.tabs(["📌 Status", "💬 Communicate", "📝 Internal Notes"])

                # ── Status Tab ──
                with action_tab1:
                    btn_c1, btn_c2, btn_c3 = st.columns(3)
                    with btn_c1:
                        if st.button("✅ Approve", key=f"approve_{exp['id']}", width="stretch", type="primary"):
                            st.session_state.expenses[exp_idx]["status"] = "Approved"
                            st.success("Expense approved! ✅")
                            st.balloons()
                            st.rerun()
                    with btn_c2:
                        if st.button("❌ Reject", key=f"reject_{exp['id']}", width="stretch"):
                            st.session_state.expenses[exp_idx]["status"] = "Rejected"
                            st.error("Expense rejected.")
                            st.rerun()
                    with btn_c3:
                        if st.button("⏳ Keep Pending", key=f"pending_{exp['id']}", width="stretch"):
                            st.session_state.expenses[exp_idx]["status"] = "Pending Approval"
                            st.info("Marked as pending.")
                            st.rerun()

                # ── Communicate Tab ──
                with action_tab2:
                    st.markdown(f"**Send message to:** {exp['employee_name']}")
                    channel = st.radio(
                        "Channel", ["📧 Email", "💬 Instant Chat"],
                        horizontal=True, key=f"channel_{exp['id']}",
                    )
                    msg_text = st.text_area(
                        "Message",
                        placeholder="e.g., Hi, could you please provide the project code for this expense?",
                        key=f"msg_{exp['id']}",
                    )
                    if st.button("📤 Send Message", key=f"send_{exp['id']}", width="stretch"):
                        if msg_text.strip():
                            st.session_state.expenses[exp_idx].setdefault("messages", []).append({
                                "channel": channel,
                                "text": msg_text,
                                "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "from": "Auditor",
                            })
                            st.success(f"✅ Message sent via {channel} to {exp['employee_name']}")
                        else:
                            st.warning("Please enter a message.")

                    # Show message history
                    msgs = exp.get("messages", [])
                    if msgs:
                        st.markdown("---")
                        st.markdown("**📜 Message History**")
                        for m in reversed(msgs):
                            st.caption(f"{m['sent_at']} — {m['channel']} — From: {m['from']}")
                            st.markdown(f"> {m['text']}")

                # ── Internal Notes Tab ──
                with action_tab3:
                    st.markdown("**📝 Log internal notes** (face-to-face discussions, phone calls, etc.)")
                    note_text = st.text_area(
                        "Notes",
                        value=exp.get("auditor_notes", ""),
                        placeholder="e.g., Spoke with employee — confirmed project code is FALCON-2026.",
                        key=f"notes_{exp['id']}",
                    )
                    follow_up = st.date_input(
                        "📅 Set Follow-up Alert / Reminder",
                        value=exp.get("follow_up_date") or date.today() + timedelta(days=3),
                        key=f"followup_{exp['id']}",
                    )
                    if st.button("💾 Save Notes", key=f"savenotes_{exp['id']}", width="stretch"):
                        st.session_state.expenses[exp_idx]["auditor_notes"] = note_text
                        st.session_state.expenses[exp_idx]["follow_up_date"] = follow_up
                        st.success(f"✅ Notes saved. Follow-up reminder set for **{follow_up}**.")


# ══════════════════════════════════════════════════════════════
# EMPLOYEE VIEW (UC1)
# ══════════════════════════════════════════════════════════════
elif "UC1" in use_case and not is_auditor:

    st.markdown(
        '<div class="hero">'
        "<h1>💳 Employee Expense Portal</h1>"
        "<p>Submit receipts, track approvals, and let AI speed up your reimbursements.</p>"
        "</div>", unsafe_allow_html=True,
    )

    tab_submit, tab_history = st.tabs(["📤 Submit New Expense", "📜 My Expense History"])

    # ────────────────────────────────────────────
    # Tab: History
    # ────────────────────────────────────────────
    with tab_history:
        st.markdown("### 📜 My Expense History")
        if not expenses:
            st.info("No expenses submitted yet.")
        else:
            hist_df = pd.DataFrame([
                {
                    "ID": e["id"],
                    "Amount": f"${e['amount']:,.2f}",
                    "Date": e["date"],
                    "Category": e.get("category", "—"),
                    "Description": e.get("description", "—"),
                    "Status": e["status"],
                }
                for e in expenses
            ])
            st.dataframe(hist_df, width="stretch", hide_index=True)

            # Status legend
            st.markdown("")
            st.markdown(
                '<span class="badge b-auto">Auto-Approved</span> '
                '<span class="badge b-approved">Approved</span> '
                '<span class="badge b-pending">Pending</span> '
                '<span class="badge b-rejected">Rejected</span> '
                '<span class="badge b-flagged">Flagged</span>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────
    # Tab: Submit New Expense
    # ────────────────────────────────────────────
    with tab_submit:
        st.markdown("### 📤 Submit New Expense")

        # ── Step A: Upload ──
        st.markdown("**Step 1:** Upload your receipt")
        uploaded = st.file_uploader(
            "Drop receipt here (PNG, JPG, PDF)",
            type=["png", "jpg", "jpeg", "pdf"],
            key="emp_upload",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()
            mime_type = get_mime(uploaded.name)

            # ── Step B: AI Extraction ──
            st.divider()
            st.markdown("**Step 2:** AI Extraction")

            col_img, col_data = st.columns([1, 1.5])

            with col_img:
                if mime_type.startswith("image"):
                    st.image(file_bytes, width="stretch", caption="Uploaded Receipt")
                else:
                    st.info("📄 PDF uploaded")

            with col_data:
                with st.spinner("🔍 Document AI is extracting data from your receipt..."):
                    # ── Real Document AI call ──
                    try:
                        extracted = extract_with_docai(file_bytes, mime_type)
                        raw_amount = extracted.get("total_amount", extracted.get("net_amount", ""))
                        raw_date = extracted.get("receipt_date", extracted.get("invoice_date", ""))
                        raw_supplier = extracted.get("supplier_name", extracted.get("supplier_address", ""))
                    except Exception as e:
                        st.error(f"Document AI Error: {e}")
                        raw_amount, raw_date, raw_supplier = "", "", ""

                # Parse amount to float
                try:
                    amount_val = float(raw_amount.replace(",", "").replace("$", "").replace("NT", "").strip())
                except (ValueError, AttributeError):
                    amount_val = 0.0

                # Editable fields
                st.markdown("##### ✏️ Verify Extracted Data")
                fc1, fc2 = st.columns(2)
                with fc1:
                    amount_input = st.number_input(
                        "💰 Amount ($)", value=amount_val, min_value=0.0, step=1.0,
                        key="amount_input",
                    )
                with fc2:
                    date_input = st.text_input("📅 Date", value=raw_date or str(date.today()), key="date_input")

                fc3, fc4 = st.columns(2)
                with fc3:
                    supplier_input = st.text_input("🏪 Supplier", value=raw_supplier or "", key="supplier_input")
                with fc4:
                    category = st.selectbox(
                        "🏷️ Category",
                        ["Meals & Entertainment", "Travel & Transportation",
                         "Equipment & Supplies", "Software & Subscriptions",
                         "Training & Education", "Other"],
                        key="category_input",
                    )

                description = st.text_input(
                    "📝 Brief Description",
                    placeholder="e.g., Team lunch for Q1 kickoff",
                    key="desc_input",
                )

            # ── Step C: Policy Check ──
            st.divider()
            st.markdown("**Step 3:** Policy Check & Submit")

            auto_limit = policies["auto_approve_limit"]
            flag_limit = policies["flag_threshold"]

            needs_justification = amount_input > flag_limit
            is_auto_approve = amount_input <= auto_limit

            if is_auto_approve:
                st.success(
                    f"✅ **Auto-Approve Eligible** — Amount (${amount_input:,.2f}) is within the "
                    f"auto-approve limit (${auto_limit:,.0f}). This will be approved instantly."
                )
            elif needs_justification:
                st.warning(
                    f"⚠️ **Justification Required** — Amount (${amount_input:,.2f}) exceeds the "
                    f"policy threshold (${flag_limit:,.0f}). Please provide a business justification below."
                )
            else:
                st.info(
                    f"📋 **Standard Review** — Amount (${amount_input:,.2f}) will be sent to the "
                    f"audit queue for manual review."
                )

            justification = ""
            if needs_justification:
                justification = st.text_area(
                    "📄 Business Justification (Required: Client name, Project code)",
                    placeholder="e.g., Client dinner with Acme Corp for Project Falcon. 4 attendees. Pre-approved by VP Sales.",
                    key="justification_input",
                )

            # Submit button — disabled if justification required but empty
            can_submit = True
            if needs_justification and not justification.strip():
                can_submit = False
                st.caption("⬆️ Please fill in the justification to enable submission.")

            st.markdown("")
            if st.button(
                "🚀 Submit Expense",
                disabled=not can_submit,
                type="primary",
                width="stretch",
                key="submit_expense",
            ):
                # Determine status
                if is_auto_approve:
                    status = "Auto-Approved"
                else:
                    status = "Pending Approval"

                new_expense = {
                    "id": gen_id(),
                    "employee_name": "You",
                    "amount": amount_input,
                    "date": date_input,
                    "category": category,
                    "description": description,
                    "status": status,
                    "justification": justification,
                    "auditor_notes": "",
                    "follow_up_date": None,
                    "image_bytes": file_bytes,
                    "mime_type": mime_type,
                    "messages": [],
                }

                st.session_state.expenses.append(new_expense)

                if is_auto_approve:
                    st.success(f"🎉 Expense **{new_expense['id']}** auto-approved! Amount: ${amount_input:,.2f}")
                    st.balloons()
                else:
                    st.success(
                        f"📨 Expense **{new_expense['id']}** submitted for review! "
                        f"Amount: ${amount_input:,.2f} — Status: **{status}**"
                    )

                st.info("💡 Switch to **Auditor** role to review this expense.")


# ══════════════════════════════════════════════════════════════
# USE CASE 2: Accounts Payable
# ══════════════════════════════════════════════════════════════
if "UC2" in use_case:

    ap_invoices = st.session_state.ap_invoices
    is_ap_manager = "Manager" in ap_role

    # ── Hero ──
    st.markdown(
        '<div class="hero">'
        "<h1>📑 Accounts Payable — Vendor Invoice Processing</h1>"
        "<p>Upload vendor invoices, auto-match to POs, and let AI validate for payment approval.</p>"
        "</div>", unsafe_allow_html=True,
    )

    # ── KPI Row ──
    pending_ap = [i for i in ap_invoices if i["status"] == "Pending Review"]
    approved_ap = [i for i in ap_invoices if "Approved" in i["status"]]
    flagged_ap = [i for i in ap_invoices if "Flagged" in i["status"]]
    total_ap = sum(i["amount"] for i in ap_invoices)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="stat-card"><p class="num">{len(ap_invoices)}</p><p class="lbl">Total Invoices</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-card"><p class="num">{len(pending_ap)}</p><p class="lbl">Pending Review</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-card"><p class="num">{len(flagged_ap)}</p><p class="lbl">Flagged</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-card"><p class="num">${total_ap:,.0f}</p><p class="lbl">Total Payable</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──
    tab_ap_queue, tab_ap_upload, tab_ap_aging = st.tabs(["📋 Invoice Queue", "📤 Upload New Invoice", "📊 Aging Report"])

    # ────────────────────────────────────────────
    # Tab: Invoice Queue
    # ────────────────────────────────────────────
    with tab_ap_queue:
        st.markdown("### 📋 All Vendor Invoices")

        ap_df = pd.DataFrame([
            {
                "ID": inv["id"],
                "Vendor": inv["vendor"],
                "Invoice #": inv["invoice_number"],
                "Amount": f"${inv['amount']:,.2f}",
                "Date": inv["date"],
                "Due Date": inv["due_date"],
                "PO #": inv.get("po_number") or "—",
                "PO Match": "✅" if inv["po_match"] else "❌",
                "Status": inv["status"],
            }
            for inv in ap_invoices
        ])
        st.dataframe(ap_df, width="stretch", hide_index=True)

        st.divider()

        # Select invoice to review
        reviewable_ap = [i for i in ap_invoices if i["status"] in ("Pending Review", "Flagged - Amount Mismatch")]
        if not reviewable_ap:
            st.markdown(
                '<div style="text-align:center;padding:2rem;color:#94a3b8;">'
                "🎉 <strong>All caught up!</strong> No invoices pending review."
                "</div>", unsafe_allow_html=True,
            )
        else:
            st.markdown("### 🔎 Review Invoice")
            ap_options = {
                f"{inv['id']} — {inv['vendor']} — ${inv['amount']:,.2f}": inv["id"]
                for inv in reviewable_ap
            }
            ap_selected_label = st.selectbox("Select an invoice to review:", list(ap_options.keys()), key="ap_select")
            ap_selected_id = ap_options[ap_selected_label]
            inv = next(i for i in ap_invoices if i["id"] == ap_selected_id)
            inv_idx = next(idx for idx, i in enumerate(ap_invoices) if i["id"] == ap_selected_id)

            st.markdown("")
            col_inv_left, col_inv_right = st.columns([1, 1.5])

            # ── Left: Invoice Details ──
            with col_inv_left:
                st.markdown("#### 🖼️ Invoice Document")
                if inv.get("image_bytes"):
                    if inv["mime_type"] and inv["mime_type"].startswith("image"):
                        st.image(inv["image_bytes"], width="stretch")
                    else:
                        st.info("📄 PDF invoice attached")
                else:
                    st.markdown(
                        '<div style="background:#f1f5f9;border-radius:12px;padding:3rem;text-align:center;color:#94a3b8;">'
                        "📄<br/>Invoice document not available<br/><span style='font-size:0.75rem;'>Demo record — no file attached</span>"
                        "</div>", unsafe_allow_html=True,
                    )

                st.markdown("")
                st.markdown("#### 📊 Invoice Data")
                st.markdown(f"""
<div class="info-block">
<strong>🏢 Vendor:</strong> {inv['vendor']}<br/>
<strong>📄 Invoice #:</strong> {inv['invoice_number']}<br/>
<strong>💰 Amount:</strong> ${inv['amount']:,.2f} {inv['currency']}<br/>
<strong>📅 Invoice Date:</strong> {inv['date']}<br/>
<strong>⏰ Due Date:</strong> {inv['due_date']}<br/>
<strong>📋 Payment Terms:</strong> {inv['payment_terms']}<br/>
<strong>🏷️ Category:</strong> {inv['category']}
</div>
                """, unsafe_allow_html=True)

                # PO Match Info
                st.markdown("#### 🔗 Purchase Order Match")
                if inv["po_number"]:
                    if inv["po_match"] and inv["amount"] == inv["po_amount"]:
                        st.success(f"✅ Matched to **{inv['po_number']}** — PO Amount: ${inv['po_amount']:,.2f} (Exact match)")
                    elif inv["po_match"]:
                        diff = inv["amount"] - inv["po_amount"]
                        st.warning(
                            f"⚠️ Matched to **{inv['po_number']}** — PO Amount: ${inv['po_amount']:,.2f}\n\n"
                            f"**Variance:** ${diff:,.2f} ({diff/inv['po_amount']*100:.1f}% over)"
                        )
                    else:
                        st.error(f"❌ PO **{inv['po_number']}** not found in system")
                else:
                    st.warning("⚠️ No Purchase Order number provided")

                # Line Items
                if inv.get("line_items"):
                    st.markdown("#### 📦 Line Items")
                    li_df = pd.DataFrame(inv["line_items"])
                    li_df["total"] = li_df["qty"] * li_df["unit_price"]
                    li_df.columns = ["Description", "Qty", "Unit Price", "Total"]
                    li_df["Unit Price"] = li_df["Unit Price"].apply(lambda x: f"${x:,.2f}")
                    li_df["Total"] = li_df["Total"].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(li_df, width="stretch", hide_index=True)

            # ── Right: AI Analysis + Actions ──
            with col_inv_right:
                st.markdown("#### 🤖 Gemini AI Invoice Analysis")

                with st.spinner("🧠 Gemini is analyzing this invoice..."):
                    line_items_str = "\n".join(
                        [f"  - {li['description']}: {li['qty']} x ${li['unit_price']:,.2f}" for li in inv.get("line_items", [])]
                    )
                    gemini_ap_prompt = f"""You are an Accounts Payable AI assistant for a corporate finance department.

Vendor Invoice Details:
- Invoice ID: {inv['id']}
- Vendor: {inv['vendor']}
- Invoice Number: {inv['invoice_number']}
- Amount: ${inv['amount']:,.2f} {inv['currency']}
- Invoice Date: {inv['date']}
- Due Date: {inv['due_date']}
- Payment Terms: {inv['payment_terms']}
- Category: {inv['category']}
- PO Number: {inv.get('po_number') or 'None'}
- PO Match: {'Yes' if inv['po_match'] else 'No'}
- PO Amount: ${inv['po_amount']:,.2f}
- Line Items:
{line_items_str}
- Existing Notes: {inv.get('notes', 'None')}

Instructions:
1. Validate the invoice: check for completeness (vendor name, invoice number, amount, dates).
2. If there is a PO, compare the invoice amount to PO amount and flag any discrepancy.
3. Check if payment terms and due date are consistent.
4. Look for potential issues: duplicate risk, unusual amounts, missing details.
5. Give a clear recommendation: APPROVE FOR PAYMENT, HOLD FOR REVIEW, or REJECT.
6. Keep your analysis to 5-6 sentences. Be professional.
"""
                    try:
                        ap_ai_analysis = call_gemini(gemini_ap_prompt)
                    except Exception as e:
                        ap_ai_analysis = f"⚠️ Gemini unavailable: {e}"

                if "reject" in ap_ai_analysis.lower():
                    st.error(f"🤖 **AI Assessment:**\n\n{ap_ai_analysis}")
                elif "hold" in ap_ai_analysis.lower() or "review" in ap_ai_analysis.lower():
                    st.warning(f"🤖 **AI Assessment:**\n\n{ap_ai_analysis}")
                else:
                    st.info(f"🤖 **AI Assessment:**\n\n{ap_ai_analysis}")

                st.divider()
                st.markdown("#### ✍️ AP Actions")

                ap_act_tab1, ap_act_tab2 = st.tabs(["📌 Status", "📝 Notes"])

                with ap_act_tab1:
                    btn_a1, btn_a2, btn_a3 = st.columns(3)
                    with btn_a1:
                        if st.button("✅ Approve for Payment", key=f"ap_approve_{inv['id']}", width="stretch", type="primary"):
                            st.session_state.ap_invoices[inv_idx]["status"] = "Approved for Payment"
                            st.success("Invoice approved for payment! ✅")
                            st.balloons()
                            st.rerun()
                    with btn_a2:
                        if st.button("🔄 Hold for Review", key=f"ap_hold_{inv['id']}", width="stretch"):
                            st.session_state.ap_invoices[inv_idx]["status"] = "On Hold"
                            st.warning("Invoice placed on hold.")
                            st.rerun()
                    with btn_a3:
                        if st.button("❌ Reject", key=f"ap_reject_{inv['id']}", width="stretch"):
                            st.session_state.ap_invoices[inv_idx]["status"] = "Rejected"
                            st.error("Invoice rejected.")
                            st.rerun()

                with ap_act_tab2:
                    ap_note = st.text_area(
                        "Notes",
                        value=inv.get("notes", ""),
                        placeholder="e.g., Confirmed with procurement — overage was pre-approved.",
                        key=f"ap_notes_{inv['id']}",
                    )
                    if st.button("💾 Save Notes", key=f"ap_savenotes_{inv['id']}", width="stretch"):
                        st.session_state.ap_invoices[inv_idx]["notes"] = ap_note
                        st.success("✅ Notes saved.")

    # ────────────────────────────────────────────
    # Tab: Upload New Invoice
    # ────────────────────────────────────────────
    with tab_ap_upload:
        st.markdown("### 📤 Upload Vendor Invoice")
        st.markdown("Upload a vendor invoice to extract data with Document AI and auto-match to Purchase Orders.")

        ap_uploaded = st.file_uploader(
            "Drop vendor invoice here (PNG, JPG, PDF)",
            type=["png", "jpg", "jpeg", "pdf"],
            key="ap_upload",
        )

        if ap_uploaded is not None:
            ap_file_bytes = ap_uploaded.read()
            ap_mime_type = get_mime(ap_uploaded.name)

            col_ap_img, col_ap_data = st.columns([1, 1.5])

            with col_ap_img:
                if ap_mime_type.startswith("image"):
                    st.image(ap_file_bytes, width="stretch", caption="Uploaded Invoice")
                else:
                    st.info("📄 PDF uploaded")

            with col_ap_data:
                with st.spinner("🔍 Document AI is extracting invoice data..."):
                    try:
                        ap_extracted = extract_with_docai(ap_file_bytes, ap_mime_type)
                    except Exception as e:
                        st.error(f"Document AI Error: {e}")
                        ap_extracted = {}

                st.markdown("##### ✏️ Verify Extracted Data")

                ac1, ac2 = st.columns(2)
                with ac1:
                    ap_vendor = st.text_input("🏢 Vendor Name",
                        value=ap_extracted.get("supplier_name", ap_extracted.get("receiver_name", "")),
                        key="ap_vendor")
                with ac2:
                    ap_inv_num = st.text_input("📄 Invoice Number",
                        value=ap_extracted.get("invoice_id", ""),
                        key="ap_inv_num")

                ac3, ac4 = st.columns(2)
                with ac3:
                    raw_ap_amount = ap_extracted.get("total_amount", ap_extracted.get("net_amount", ""))
                    try:
                        ap_amount_val = float(str(raw_ap_amount).replace(",", "").replace("$", "").replace("NT", "").strip())
                    except (ValueError, AttributeError):
                        ap_amount_val = 0.0
                    ap_amount = st.number_input("💰 Invoice Amount ($)", value=ap_amount_val, min_value=0.0, step=1.0, key="ap_amount")
                with ac4:
                    ap_currency = st.selectbox("💱 Currency", ["USD", "TWD", "EUR", "GBP", "JPY"], key="ap_currency")

                ac5, ac6 = st.columns(2)
                with ac5:
                    raw_ap_date = ap_extracted.get("invoice_date", ap_extracted.get("receipt_date", ""))
                    ap_date = st.text_input("📅 Invoice Date", value=raw_ap_date or str(date.today()), key="ap_date")
                with ac6:
                    ap_due = st.text_input("⏰ Due Date", value=ap_extracted.get("due_date", ""), key="ap_due")

                ac7, ac8 = st.columns(2)
                with ac7:
                    ap_po = st.text_input("📋 PO Number (if applicable)", value="", key="ap_po")
                with ac8:
                    ap_cat = st.selectbox(
                        "🏷️ Category",
                        ["IT Hardware", "Cloud Services", "Office Supplies", "Professional Services",
                         "Marketing", "Facilities", "Other"],
                        key="ap_cat",
                    )

                ap_terms = st.selectbox("📋 Payment Terms", ["Net 30", "Net 45", "Net 60", "Due on Receipt"], key="ap_terms")

            st.divider()

            # PO Match simulation
            if ap_po:
                existing_pos = {inv["po_number"]: inv["po_amount"] for inv in ap_invoices if inv["po_number"]}
                if ap_po in existing_pos:
                    po_amt = existing_pos[ap_po]
                    if abs(ap_amount - po_amt) < 0.01:
                        st.success(f"✅ **PO Match Found** — {ap_po} for ${po_amt:,.2f} (Exact match)")
                        ap_po_match = True
                        ap_po_amount = po_amt
                    else:
                        diff = ap_amount - po_amt
                        st.warning(f"⚠️ **PO Found but Amount Mismatch** — {ap_po} for ${po_amt:,.2f} (Variance: ${diff:,.2f})")
                        ap_po_match = True
                        ap_po_amount = po_amt
                else:
                    st.info(f"📋 PO **{ap_po}** will be recorded (not yet in system for matching)")
                    ap_po_match = False
                    ap_po_amount = 0.0
            else:
                st.caption("No PO number provided — invoice will require manual approval.")
                ap_po_match = False
                ap_po_amount = 0.0

            st.markdown("")
            if st.button("🚀 Submit Invoice", type="primary", width="stretch", key="submit_ap_invoice"):
                new_inv = {
                    "id": f"INV-{uuid.uuid4().hex[:3].upper()}",
                    "vendor": ap_vendor or "Unknown Vendor",
                    "invoice_number": ap_inv_num or "N/A",
                    "amount": ap_amount,
                    "currency": ap_currency,
                    "date": ap_date,
                    "due_date": ap_due or "N/A",
                    "po_number": ap_po,
                    "po_match": ap_po_match,
                    "po_amount": ap_po_amount,
                    "category": ap_cat,
                    "payment_terms": ap_terms,
                    "status": "Pending Review",
                    "line_items": [],
                    "notes": "",
                    "image_bytes": ap_file_bytes,
                    "mime_type": ap_mime_type,
                }
                st.session_state.ap_invoices.append(new_inv)
                st.success(f"📨 Invoice **{new_inv['id']}** from **{new_inv['vendor']}** submitted for review! Amount: ${ap_amount:,.2f}")

    # ────────────────────────────────────────────
    # Tab: Aging Report
    # ────────────────────────────────────────────
    with tab_ap_aging:
        st.markdown("### 📊 Accounts Payable Aging Report")
        st.markdown("Overview of outstanding payables by aging bucket.")

        today = date.today()
        aging_data = {"Current (0-30)": 0.0, "31-60 Days": 0.0, "61-90 Days": 0.0, "90+ Days": 0.0}

        for inv in ap_invoices:
            if "Rejected" in inv["status"]:
                continue
            try:
                due = datetime.strptime(inv["due_date"], "%Y-%m-%d").date()
                days_outstanding = (today - due).days
            except (ValueError, TypeError):
                days_outstanding = 0

            if days_outstanding <= 0:
                aging_data["Current (0-30)"] += inv["amount"]
            elif days_outstanding <= 60:
                aging_data["31-60 Days"] += inv["amount"]
            elif days_outstanding <= 90:
                aging_data["61-90 Days"] += inv["amount"]
            else:
                aging_data["90+ Days"] += inv["amount"]

        aging_df = pd.DataFrame([
            {"Aging Bucket": k, "Amount": f"${v:,.2f}", "Raw": v}
            for k, v in aging_data.items()
        ])

        ac1, ac2, ac3, ac4 = st.columns(4)
        for col, (bucket, amount) in zip([ac1, ac2, ac3, ac4], aging_data.items()):
            color = "#16a34a" if "Current" in bucket else "#f59e0b" if "31" in bucket else "#ef4444" if "90+" in bucket else "#f97316"
            with col:
                st.markdown(
                    f'<div class="stat-card"><p class="num" style="color:{color}">${amount:,.0f}</p>'
                    f'<p class="lbl">{bucket}</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.bar_chart(
            pd.DataFrame({"Amount": aging_data.values()}, index=aging_data.keys()),
        )

        st.divider()
        st.markdown("#### 📋 All Invoices by Due Date")
        due_df = pd.DataFrame([
            {
                "Vendor": inv["vendor"],
                "Invoice #": inv["invoice_number"],
                "Amount": f"${inv['amount']:,.2f}",
                "Due Date": inv["due_date"],
                "Status": inv["status"],
            }
            for inv in sorted(ap_invoices, key=lambda x: x["due_date"])
        ])
        st.dataframe(due_df, width="stretch", hide_index=True)
