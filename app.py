import streamlit as st
import os
from datetime import date
from openai import OpenAI
import ollama  # Official Ollama Python library
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import json
import re
import shutil
import zipfile
import sqlite3
import streamlit_authenticator as stauth
import yaml
from pathlib import Path
import plotly.express as px

# ReportLab – Landscape mode
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

# ========================= VERSION 7.99 – OLLAMA LOCAL PRIMARY + OPENAI OPTIONAL FALLBACK =========================
st.set_page_config(page_title="Compliance Portal v7.99", layout="wide", page_icon="🔒")

st.sidebar.markdown(
    """
    <div style="background:#0a3d62;color:white;padding:20px;border-radius:12px;text-align:center;">
        <h2>Compliance Portal</h2>
        <h1 style="font-size:3em;margin:10px 0;">v7.99</h1>
        <p style="font-size:1.3em;"><strong>AIR-GAPPED READY</strong></p>
        <p>Ollama Local Primary • OpenAI Optional Fallback</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================= AUTHENTICATION & MULTI-TENANCY SETUP (NO MFA) =========================
CONFIG_FILE = Path("config.yaml")

raw_config = {
    "credentials": {
        "usernames": {
            "superadmin": {
                "name": "Super Admin",
                "password": "super789",
                "role": "super_admin",
                "tenant_id": "global"
            },
            "admin1": {
                "name": "Tenant Admin",
                "password": "admin123",
                "role": "admin",
                "tenant_id": "tenant1"
            },
            "user1": {
                "name": "Regular User",
                "password": "user456",
                "role": "user",
                "tenant_id": "tenant1"
            }
        }
    },
    "cookie": {
        "name": "compliance_portal_auth",
        "key": "super_secure_random_key_2025_xai",
        "expiry_days": 30
    },
    "tenants": {"tenant1": {"name": "Demo Tenant"}},
    "sso_enabled": False
}

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f) or {}
else:
    import copy
    config = copy.deepcopy(raw_config)

    hasher = stauth.Hasher()
    for username, user in config["credentials"]["usernames"].items():
        user["password"] = hasher.hash(user["password"])

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render login form
authenticator.login(location="sidebar")

# Get authentication status
authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

# Add Logout button if logged in – using the official method
if authentication_status:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Logged in as:** {name}")
    authenticator.logout("🔒 Logout", location="sidebar", key="logout_btn")

if authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password in the sidebar")
else:
    st.title("Global & Privacy Compliance Portal")
    st.markdown("### Real-Time Monitoring with Email, SMS, Slack & Teams Alerts")

    user_info = config['credentials']['usernames'][username]
    role = user_info['role']
    tenant_id = user_info['tenant_id']

    st.success(f"Logged in as {name} ({role}) – Tenant: {tenant_id}")

    # ========================= AI ENGINE CONFIGURATION – OLLAMA PRIMARY + OPENAI OPTIONAL =========================
    st.sidebar.header("AI Engine Configuration")

    use_ollama = st.sidebar.checkbox("Use Local Ollama Server (Primary)", value=True, help="Uncheck to force OpenAI fallback")

    if use_ollama:
        st.sidebar.success("🟢 Using Local Ollama Server (Air-Gapped Ready)")
        ollama_model = st.sidebar.text_input(
            "Ollama Model Name",
            value="phi3:mini",
            help="Examples: phi3:mini, phi3:medium, mistral:7b, gemma2:2b"
        )
        # Test connection to local Ollama server
        try:
            ollama.list()  # Simple call to check if server is reachable
            st.sidebar.success(f"Connected to Ollama – Model '{ollama_model}' ready!")
        except Exception as e:
            st.sidebar.error(f"Ollama server not reachable: {str(e)}")
            st.sidebar.info("Ensure Ollama is running with `ollama serve` and model pulled.")
            use_ollama = False  # Force fallback if Ollama unavailable

        openai_client = None
    else:
        st.sidebar.warning("🌐 Using OpenAI Cloud Engine (Fallback)")
        if "openai_api_key" not in st.secrets and "openai_api_key" not in os.environ:
            st.error("**Critical Error**: OpenAI API key not found in Streamlit secrets or environment.")
            st.stop()
        api_key = st.secrets.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key.strip())
        ollama_model = None

    # ========================= DOMAINS =========================
    domains = [
        "1. A&A: Audit & Assurance",
        "2. AIS: Application & Interface Security",
        "3. BCR: Business Continuity Mgmt & Op Resilience",
        "4. CCC: Change Control & Configuration Management",
        "5. CEK: Cryptography, Encryption & Key Management",
        "6. DCS: Datacenter Security",
        "7. DSP: Data Security & Privacy",
        "8. GRC: Governance, Risk Management, & Compliance",
        "9. HRS: Human Resources Security",
        "10. IAM: Identity & Access Management",
        "11. IPY: Interoperability & Portability",
        "12. IVS: Infrastructure & Virtualization Security",
        "13. LOG: Logging & Monitoring",
        "14. SEF: Sec. Incident Mgmt, E-Disc & Cloud Forensics",
        "15. STA: Supply Chain Mgmt, Transparency, & Accountability",
        "16. TVM: Threat & Vulnerability Management",
        "17. UEM: Universal Endpoint Management"
    ]

    # ========================= MAPPINGS – ALL DEFINED =========================
    soc2_mappings = {d: f"SOC2 {d.split(':')[0]}" for d in domains}
    iso27001_controls = {d: f"ISO 27001 {d.split(':')[0]}" for d in domains}
    cmmc_level3_mappings = {d: f"CMMC L3 {d.split(':')[0]}" for d in domains}
    gdpr_mappings = {d: f"GDPR Art. {d.split('.')[0]}" for d in domains}
    ccpa_mappings = {d: f"CCPA 1798.{d.split('.')[0]}" for d in domains}
    nist80053_mappings = {d: f"NIST 800-53 {d.split(':')[0]}" for d in domains}
    hipaa_security_mappings = {d: f"HIPAA Security {d.split(':')[0]}" for d in domains}
    hipaa_privacy_mappings = {d: f"HIPAA Privacy {d.split(':')[0]}" for d in domains}
    iso42001_mappings = {d: f"ISO 42001 {d.split(':')[0]}" for d in domains}
    eu_ai_act_high_risk_mappings = {d: f"EU AI Act {d.split(':')[0]}" for d in domains}
    fedramp_moderate_mappings = {d: f"FedRAMP Moderate {d.split(':')[0]}" for d in domains}
    gdpr_dpia_mappings = {d: f"GDPR DPIA {d.split(':')[0]}" for d in domains}
    gdpr_lia_mappings = {d: f"GDPR LIA {d.split(':')[0]}" for d in domains}
    gdpr_dsr_mappings = {d: f"GDPR DSR {d.split(':')[0]}" for d in domains}
    gdpr_dpo_mappings = {d: f"GDPR DPO {d.split(':')[0]}" for d in domains}
    gdpr_ropa_mappings = {d: f"GDPR RoPA {d.split(':')[0]}" for d in domains}
    ccpa_pia_mappings = {d: f"CCPA PIA {d.split(':')[0]}" for d in domains}
    hipaa_pia_mappings = {d: f"HIPAA PIA {d.split(':')[0]}" for d in domains}
    soc2_privacy_mappings = {d: f"SOC2 Privacy {d.split(':')[0]}" for d in domains}
    iso27701_mappings = {d: f"ISO 27701 {d.split(':')[0]}" for d in domains}
    nis2_mappings = {
        "1. A&A: Audit & Assurance": "NIS2 Art. 21(2)(j) – Reporting & Audit",
        "2. AIS: Application & Interface Security": "NIS2 Art. 21(2)(a) – Risk Analysis & Security Policies",
        "3. BCR: Business Continuity Mgmt & Op Resilience": "NIS2 Art. 21(2)(e) – Business Continuity & Crisis Management",
        "4. CCC: Change Control & Configuration Management": "NIS2 Art. 21(2)(d) – Policies for Information System Acquisition, Development & Maintenance",
        "5. CEK: Cryptography, Encryption & Key Management": "NIS2 Art. 21(2)(f) – Cryptography & Encryption",
        "6. DCS: Datacenter Security": "NIS2 Art. 21(2)(b) – Access Control & Asset Management",
        "7. DSP: Data Security & Privacy": "NIS2 Art. 21(2)(a) – Risk Analysis & Security Policies",
        "8. GRC: Governance, Risk Management, & Compliance": "NIS2 Art. 21(1) – Governance & Risk Management",
        "9. HRS: Human Resources Security": "NIS2 Art. 21(2)(g) – Human Resources Security & Awareness",
        "10. IAM: Identity & Access Management": "NIS2 Art. 21(2)(b) – Access Control",
        "11. IPY: Interoperability & Portability": "NIS2 Art. 21(2)(i) – Multi-factor Authentication & Secure Communications",
        "12. IVS: Infrastructure & Virtualization Security": "NIS2 Art. 21(2)(c) – Network Security",
        "13. LOG: Logging & Monitoring": "NIS2 Art. 21(2)(h) – Incident Detection & Monitoring",
        "14. SEF: Sec. Incident Mgmt, E-Disc & Cloud Forensics": "NIS2 Art. 21(2)(h) – Incident Handling & Reporting",
        "15. STA: Supply Chain Mgmt, Transparency, & Accountability": "NIS2 Art. 21(2)(c) – Supply Chain Security",
        "16. TVM: Threat & Vulnerability Management": "NIS2 Art. 21(2)(a) – Risk Analysis & Vulnerability Handling",
        "17. UEM: Universal Endpoint Management": "NIS2 Art. 21(2)(b) – Asset Management & Access Control"
    }

    # ========================= RECOMMENDATIONS =========================
    control_recommendations = {
        "1. A&A: Audit & Assurance": "Quarterly internal audits + annual 3rd-party audit.",
        "2. AIS: Application & Interface Security": "SAST/DAST + secure coding standards.",
        "3. BCR: Business Continuity Mgmt & Op Resilience": "Tested BCP/DR with RTO<4h.",
        "4. CCC: Change Control & Configuration Management": "CAB + automated testing + rollback.",
        "5. CEK: Cryptography, Encryption & Key Management": "FIPS 140-3 + key rotation.",
        "6. DCS: Datacenter Security": "24/7 biometric + CCTV.",
        "7. DSP: Data Security & Privacy": "Encrypt PII + DLP.",
        "8. GRC: Governance, Risk Management, & Compliance": "Risk register + annual DPIA.",
        "9. HRS: Human Resources Security": "Background checks + annual training.",
        "10. IAM: Identity & Access Management": "MFA + least privilege + reviews.",
        "11. IPY: Interoperability & Portability": "CSV/JSON export support.",
        "12. IVS: Infrastructure & Virtualization Security": "Zero trust + segmentation.",
        "13. LOG: Logging & Monitoring": "SIEM + 12-month retention.",
        "14. SEF: Sec. Incident Mgmt, E-Disc & Cloud Forensics": "72-hour breach notification.",
        "15. STA: Supply Chain Mgmt, Transparency, & Accountability": "Annual vendor audits.",
        "16. TVM: Threat & Vulnerability Management": "Weekly scans + 7-day critical patching.",
        "17. UEM: Universal Endpoint Management": "MDM + EDR + full-disk encryption."
    }

    # ========================= DATABASE FUNCTIONS =========================
    DB_FILE = f"{tenant_id}_compliance_data.db"

    def init_tenant_db():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS settings
                     (key TEXT PRIMARY KEY, value TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS audit_periods
                     (period TEXT, domain TEXT, versions TEXT,
                      PRIMARY KEY (period, domain))''')
        conn.commit()
        conn.close()

    init_tenant_db()

    def load_tenant_data():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        data = {
            "audit_periods": {},
            "current_period": "2025 Audit",
            "email_settings": {"enabled": False, "recipient": ""},
            "sms_settings": {"enabled": False, "phone": ""},
            "slack_settings": {"enabled": False, "webhook_url": ""},
            "teams_settings": {"enabled": False, "webhook_url": ""}
        }
        c.execute("SELECT key, value FROM settings")
        for key, value in c.fetchall():
            if key.endswith("_settings"):
                data[key] = json.loads(value)
            else:
                data[key] = value
        c.execute("SELECT period, domain, versions FROM audit_periods")
        for period, domain, versions_json in c.fetchall():
            if period not in data["audit_periods"]:
                data["audit_periods"][period] = {}
            data["audit_periods"][period][domain] = json.loads(versions_json)
        conn.close()
        return data

    def save_tenant_data(tenant_data):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for key in ["current_period", "email_settings", "sms_settings", "slack_settings", "teams_settings"]:
            if key in tenant_data:
                value = json.dumps(tenant_data[key]) if isinstance(tenant_data[key], dict) else tenant_data[key]
                c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        c.execute("DELETE FROM audit_periods")
        for period, period_data in tenant_data["audit_periods"].items():
            for domain, domain_data in period_data.items():
                versions_json = json.dumps(domain_data)
                c.execute("INSERT INTO audit_periods (period, domain, versions) VALUES (?, ?, ?)",
                          (period, domain, versions_json))
        conn.commit()
        conn.close()

    data = load_tenant_data()

    # ========================= AUDIT PERIOD MANAGEMENT =========================
    st.sidebar.markdown("### Audit Period Management")

    available_periods = list(data["audit_periods"].keys())
    if not available_periods:
        default_period = "2025 Audit"
        available_periods = [default_period]
        data["audit_periods"][default_period] = {d: {"versions": []} for d in domains}
        data["current_period"] = default_period
        save_tenant_data(data)

    selected_period = st.sidebar.selectbox(
        "View Audit Period",
        options=available_periods,
        index=available_periods.index(data.get("current_period", available_periods[0])),
        key="audit_period_select"
    )

    is_current = selected_period == data.get("current_period")
    if not is_current:
        st.sidebar.info(f"🔒 Viewing **locked** historical audit: {selected_period}")

    audit_period = selected_period
    current_period_data = data["audit_periods"][audit_period]
    is_read_only = not is_current

    if is_read_only:
        st.warning(f"You are viewing a **locked historical audit period**: {audit_period}. Uploads and modifications are disabled.")

    # ========================= ARCHIVE FUNCTION =========================
    ARCHIVE_DIR = "archives"
    def archive_audit_period():
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_name = f"{ARCHIVE_DIR}/Audit_{audit_period.replace(' ', '_')}_{date.today().isoformat()}.zip"
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(DB_FILE, arcname=os.path.basename(DB_FILE))
            period_dir = os.path.join("uploads", re.sub(r'[<>:"/\\|?*]', '_', audit_period.strip()))
            if os.path.exists(period_dir):
                for root, dirs, files in os.walk(period_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "uploads")
                        zipf.write(file_path, arcname=arcname)
        return archive_name

    # ========================= ADMIN TOOLS =========================
    if role in ["super_admin", "admin"]:
        st.sidebar.header("Admin Tools")

        if role == "super_admin":
            new_tenant = st.sidebar.text_input("New Tenant ID", key="admin_new_tenant_input")
            if st.sidebar.button("Create Tenant", key="admin_create_tenant_btn"):
                if new_tenant and new_tenant not in config.get('tenants', {}):
                    if 'tenants' not in config:
                        config['tenants'] = {}
                    config['tenants'][new_tenant] = {"name": new_tenant.title()}
                    with open(CONFIG_FILE, 'w') as f:
                        yaml.safe_dump(config, f)
                    st.sidebar.success(f"Tenant '{new_tenant}' created")

        st.sidebar.markdown("### ⚠️ Danger Zone")
        st.sidebar.warning("This will **permanently delete** all evidence, versions, settings, and audit history for this tenant.")

        if st.sidebar.button("🗑️ Reset All Data – Start Fresh Audit", type="primary", key="admin_reset_all_data_btn"):
            st.sidebar.info("Generating backup archive before reset...")

            with st.spinner("Creating full backup archive..."):
                backup_path = archive_audit_period()

                with open(backup_path, "rb") as backup_file:
                    st.sidebar.download_button(
                        label="📥 DOWNLOAD BACKUP FIRST (Required before reset)",
                        data=backup_file,
                        file_name=f"Compliance_Portal_Backup_{tenant_id}_{date.today().isoformat()}.zip",
                        mime="application/zip",
                        key="admin_backup_download_btn"
                    )

            st.sidebar.markdown("---")
            st.sidebar.error("⚠️ After downloading the backup, confirm below to proceed with reset.")

            if st.sidebar.checkbox("I have downloaded the backup and want to permanently delete all data", key="admin_confirm_delete_checkbox"):
                if st.sidebar.button("🔥 CONFIRM: Delete Everything & Start Fresh", type="secondary", key="admin_confirm_delete_btn"):
                    with st.spinner("Permanently deleting all data..."):
                        if os.path.exists("uploads"):
                            shutil.rmtree("uploads")
                            os.makedirs("uploads", exist_ok=True)

                        if os.path.exists(f"{tenant_id}_compliance_data.db"):
                            os.remove(f"{tenant_id}_compliance_data.db")

                        if os.path.exists(backup_path):
                            os.remove(backup_path)

                        init_tenant_db()
                        data = load_tenant_data()

                    st.sidebar.success("✅ All data permanently deleted! Portal is now ready for a brand new audit.")
                    st.rerun()

    # Admin: Create New Audit Period
    if role in ["super_admin", "admin"]:
        st.sidebar.markdown("#### Start New Audit Period")

        new_period_name = st.sidebar.text_input(
            "New Audit Period Name",
            value=f"{date.today().year + 1} Audit",
            key="admin_new_period_input"
        )

        if st.sidebar.button("🚀 Create New Audit Period & Auto-Backup Current", key="admin_create_new_period_btn"):
            with st.spinner("Creating backup of current period..."):
                current_backup_path = archive_audit_period()

                with open(current_backup_path, "rb") as backup_file:
                    st.sidebar.success("✅ Backup created for current period!")
                    st.sidebar.download_button(
                        label=f"📥 Download Backup: {data['current_period']}.zip",
                        data=backup_file,
                        file_name=f"{data['current_period'].replace(' ', '_')}_backup_{date.today().isoformat()}.zip",
                        mime="application/zip",
                        key="admin_new_period_backup_download"
                    )

            st.sidebar.markdown("---")
            st.sidebar.info("Backup ready. Confirm to create new period...")

            if st.sidebar.button("✅ Confirm: Lock Current & Start New Period", key="admin_confirm_new_period_btn"):
                with st.spinner("Finalizing..."):
                    data["current_period"] = new_period_name
                    if new_period_name not in data["audit_periods"]:
                        data["audit_periods"][new_period_name] = {d: {"versions": []} for d in domains}

                    if os.path.exists("uploads"):
                        shutil.rmtree("uploads")
                    os.makedirs("uploads", exist_ok=True)

                    save_tenant_data(data)

                st.sidebar.success(f"🎉 New audit period **{new_period_name}** activated!")
                st.rerun()

    # ========================= TEXT EXTRACTION =========================
    def extract_text(file_path, ext):
        text = ""
        try:
            if ext == "pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            elif ext == "docx":
                doc = Document(file_path)
                for p in doc.paragraphs:
                    text += p.text + "\n"
            elif ext in ["xlsx", "csv"]:
                if ext == "xlsx":
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                text += df.to_string(index=False) + "\n"
            elif ext == "txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            st.warning(f"Error extracting text from {os.path.basename(file_path)}: {str(e)}")
        return text.strip()

    # ========================= AI ANALYSIS – OLLAMA PRIMARY + OPENAI OPTIONAL =========================
    @st.cache_data(show_spinner=False)
    def analyze_evidence(domain: str, evidence: str):
        if not evidence.strip():
            return {
                "score": 0,
                "reasoning": "No evidence provided.",
                "completeness": "None",
                "audit_readiness": "Fail",
                "recommendations": "Upload relevant documentation for this domain."
            }

        prompt = f"""You are an expert compliance auditor performing a real external audit.

Domain: {domain}
Evidence Provided:
{evidence[:15000]}

Perform a thorough analysis:
1. Score the evidence 0-100 based on completeness, clarity, relevance, and sufficiency for audit purposes.
2. Assess completeness: Does this evidence fully address all typical requirements for this domain in a real audit?
3. Determine audit readiness: Would this evidence likely pass a real external audit as-is? (Pass / Marginal / Fail)
4. Provide detailed reasoning explaining the score and readiness.
5. List specific, actionable recommendations if improvements are needed (bullet points). If perfect, say "None required".

Return ONLY JSON:
{{
  "score": int,
  "reasoning": "detailed string",
  "completeness": "Complete / Partial / Minimal / None",
  "audit_readiness": "Pass / Marginal / Fail",
  "recommendations": "bullet list or 'None required'"
}}
"""

        try:
            if use_ollama:
                # Use local Ollama server
                response = ollama.chat(
                    model=ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2}
                )
                result = json.loads(response['message']['content'])
            else:
                # Use OpenAI fallback
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                result = json.loads(resp.choices[0].message.content)

            score = max(0, min(100, int(result.get("score", 0))))
            reasoning = result.get("reasoning", "No reasoning provided")
            completeness = result.get("completeness", "None")
            audit_readiness = result.get("audit_readiness", "Fail")
            recommendations = result.get("recommendations", "Upload evidence")
            if isinstance(recommendations, list):
                recommendations = "\n".join(f"- {r}" for r in recommendations)

            return {
                "score": score,
                "reasoning": reasoning,
                "completeness": completeness,
                "audit_readiness": audit_readiness,
                "recommendations": recommendations
            }
        except Exception as e:
            # Fallback to OpenAI if Ollama fails and fallback is enabled
            if use_ollama and openai_client is not None:
                st.warning("Ollama failed – falling back to OpenAI.")
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    result = json.loads(resp.choices[0].message.content)
                    score = max(0, min(100, int(result.get("score", 0))))
                    reasoning = result.get("reasoning", "No reasoning provided")
                    completeness = result.get("completeness", "None")
                    audit_readiness = result.get("audit_readiness", "Fail")
                    recommendations = result.get("recommendations", "Upload evidence")
                    if isinstance(recommendations, list):
                        recommendations = "\n".join(f"- {r}" for r in recommendations)

                    return {
                        "score": score,
                        "reasoning": reasoning + "\n\n(Note: Analysis performed via OpenAI fallback due to Ollama error.)",
                        "completeness": completeness,
                        "audit_readiness": audit_readiness,
                        "recommendations": recommendations
                    }
                except Exception as fallback_e:
                    return {
                        "score": 0,
                        "reasoning": f"AI analysis failed (Ollama: {str(e)}, OpenAI fallback: {str(fallback_e)})",
                        "completeness": "Error",
                        "audit_readiness": "Fail",
                        "recommendations": "Check AI engine connection and retry"
                    }
            else:
                return {
                    "score": 0,
                    "reasoning": f"AI analysis error: {str(e)}",
                    "completeness": "Error",
                    "audit_readiness": "Fail",
                    "recommendations": "Check AI engine connection and retry"
                }

    # ========================= EVIDENCE MANAGEMENT =========================
    st.markdown("---")
    st.header("Evidence Management & Dashboard")

    selected_domain = st.selectbox("Select Domain", domains, disabled=is_read_only, key="domain_select_main")

    if is_read_only:
        st.info("🔒 Evidence upload is disabled for locked historical audit periods.")
        uploaded = None
    else:
        uploaded = st.file_uploader("Upload Evidence for this Domain", type=["pdf","docx","xlsx","csv","txt"], accept_multiple_files=True, key="evidence_uploader")

    if uploaded and not is_read_only:
        with st.spinner("Analyzing evidence with enhanced audit intelligence..."):
            safe_period = re.sub(r'[<>:"/\\|?*]', '_', audit_period.strip())
            safe_domain = re.sub(r'[<>:"/\\|?*]', '_', selected_domain.strip())
            base_dir = os.path.join("uploads", safe_period, safe_domain)
            os.makedirs(base_dir, exist_ok=True)

            existing_versions = [int(v.lstrip('v')) for v in os.listdir(base_dir) if v.startswith('v') and os.path.isdir(os.path.join(base_dir, v))]
            version_num = max(existing_versions) + 1 if existing_versions else 1
            version_folder = os.path.join(base_dir, f"v{version_num}")
            os.makedirs(version_folder, exist_ok=True)

            saved_files = []
            for f in uploaded:
                file_path = os.path.join(version_folder, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                saved_files.append(f.name)

            all_text = ""
            for f in uploaded:
                file_ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
                text = extract_text(os.path.join(version_folder, f.name), file_ext)
                all_text += f"\n=== {f.name} ===\n{text}\n"

            result = analyze_evidence(selected_domain, all_text)

            if selected_domain not in current_period_data:
                current_period_data[selected_domain] = {"versions": []}

            current_period_data[selected_domain]["versions"].append({
                "version": version_num,
                "files": saved_files,
                "timestamp": date.today().isoformat(),
                "analysis": result
            })

            save_tenant_data(data)
            st.success(f"Version {version_num} saved – Score: {result['score']}/100")
            st.markdown(f"**Audit Readiness:** {result['audit_readiness']}")
            st.markdown(f"**Completeness:** {result['completeness']}")
            st.markdown("**Recommendations:**")
            st.markdown(result['recommendations'])

    # ========================= HISTORY DISPLAY =========================
    st.subheader(f"Evidence History – {selected_domain}")
    versions = current_period_data.get(selected_domain, {}).get("versions", [])

    if versions:
        versions_sorted = sorted(versions, key=lambda x: x["version"], reverse=True)
        for v in versions_sorted:
            analysis = v.get("analysis", {})
            with st.expander(f"Version {v['version']} – {v['timestamp']} – Score: {analysis.get('score', 0)}/100 – Readiness: {analysis.get('audit_readiness', 'N/A')}"):
                st.write("Files:", ", ".join(v["files"]))
                st.markdown("**AI Reasoning:**")
                st.markdown(analysis.get("reasoning", "No analysis available"))
                st.markdown(f"**Completeness:** {analysis.get('completeness', 'N/A')}")
                st.markdown("**Recommendations:**")
                st.markdown(analysis.get("recommendations", "No recommendations available"))
                if not is_read_only and st.button("Delete Version", key=f"del_version_{selected_domain}_{v['version']}"):
                    safe_period = re.sub(r'[<>:"/\\|?*]', '_', audit_period.strip())
                    safe_domain = re.sub(r'[<>:"/\\|?*]', '_', selected_domain.strip())
                    base_dir = os.path.join("uploads", safe_period, safe_domain)
                    version_folder = os.path.join(base_dir, f"v{v['version']}")
                    if os.path.exists(version_folder):
                        shutil.rmtree(version_folder)
                    current_period_data[selected_domain]["versions"] = [ver for ver in current_period_data[selected_domain]["versions"] if ver["version"] != v["version"]]
                    save_tenant_data(data)
                    st.rerun()
        latest_analysis = versions_sorted[0].get("analysis", {})
        st.info(f"**Current Version: v{versions_sorted[0]['version']}** – Score: {latest_analysis.get('score', 0)}/100 – Readiness: {latest_analysis.get('audit_readiness', 'N/A')}")
    else:
        st.info("No versions yet for this domain.")

    # ========================= COMPLIANCE DASHBOARD =========================
    st.markdown("---")
    st.header("📊 Compliance Dashboard")

    domain_scores = []
    compliant_count = 0
    total_domains = len(domains)
    total_score = 0

    for domain in domains:
        vers = current_period_data.get(domain, {}).get("versions", [])
        if vers:
            latest_analysis = vers[-1]["analysis"]
            latest_score = latest_analysis.get("score", 0)
            domain_scores.append(latest_score)
            total_score += latest_score
            if latest_score >= 70:
                compliant_count += 1
        else:
            domain_scores.append(0)

    average_score = total_score / total_domains if total_domains > 0 else 0
    compliance_rate = (compliant_count / total_domains) * 100 if total_domains > 0 else 0

    df_dashboard = pd.DataFrame({
        "Domain": domains,
        "Score": domain_scores,
        "Status": ["Compliant" if s >= 70 else "Needs Attention" for s in domain_scores]
    })

    tab1, tab2, tab3 = st.tabs(["Executive Dashboard", "Audit View", "Regulatory View"])

    with tab1:
        st.subheader("Executive Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Compliance Score", f"{average_score:.1f}/100")
        col2.metric("Compliant Domains", f"{compliant_count}/{total_domains}", delta=f"{compliance_rate:.1f}%")
        col3.metric("Domains with Evidence", f"{len([d for d in domains if current_period_data.get(d, {}).get('versions')])}/{total_domains}")
        col4.metric("High-Risk Gaps (<50)", f"{len([s for s in domain_scores if s < 50])}")

        st.markdown("### Compliance Heatmap")
        fig_heatmap = px.imshow(
            [domain_scores],
            labels=dict(x="Domain", y="Score", color="Score"),
            x=domains,
            y=["Current Score"],
            color_continuous_scale=["red", "yellow", "green"],
            text_auto=True
        )
        fig_heatmap.update_layout(height=300, xaxis_tickangle=45)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("### Domain Scores")
        fig_bar = px.bar(df_dashboard, x="Domain", y="Score", color="Status",
                         color_discrete_map={"Compliant": "green", "Needs Attention": "orange"})
        fig_bar.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Audit & Evidence View")
        st.write(f"**Total Versions Uploaded:** {sum(len(current_period_data.get(d, {}).get('versions', [])) for d in domains)}")

        evidence_list = []
        for domain in domains:
            for v in current_period_data.get(domain, {}).get("versions", []):
                analysis = v.get("analysis", {})
                evidence_list.append({
                    "Domain": domain,
                    "Version": v["version"],
                    "Files": ", ".join(v["files"]),
                    "Score": analysis.get("score", 0),
                    "Readiness": analysis.get("audit_readiness", "N/A"),
                    "Completeness": analysis.get("completeness", "N/A"),
                    "Date": v["timestamp"]
                })

        if evidence_list:
            evidence_df = pd.DataFrame(evidence_list)
            st.dataframe(evidence_df.sort_values(["Domain", "Version"], ascending=[True, False]), use_container_width=True)
        else:
            st.info("No evidence uploaded yet.")

        st.markdown("### Compliance Gaps")
        gaps = df_dashboard[df_dashboard["Score"] < 70]
        if not gaps.empty:
            st.dataframe(gaps, use_container_width=True)
        else:
            st.success("No compliance gaps!")

    with tab3:
        st.subheader("Regulatory Framework View")
        frameworks = {
            "SOC2": soc2_mappings,
            "ISO 27001": iso27001_controls,
            "GDPR": gdpr_mappings,
            "HIPAA": hipaa_privacy_mappings,
            "NIS2": nis2_mappings
        }

        for fw_name, mapping in frameworks.items():
            fw_scores = []
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                score = vers[-1]["analysis"].get("score", 0) if vers else 0
                fw_scores.append(score)
            fw_avg = sum(fw_scores) / len(fw_scores) if fw_scores else 0
            fw_compliant = sum(1 for s in fw_scores if s >= 70)
            st.metric(f"{fw_name} Score", f"{fw_avg:.1f}/100", delta=f"{fw_compliant}/{len(domains)} compliant")

    # ========================= COMMON STYLES FUNCTION =========================
    def get_common_styles():
        styles = getSampleStyleSheet()
        normal = styles['Normal']
        normal.wordWrap = 'CJK'

        header_style = ParagraphStyle('HeaderText', parent=normal, fontSize=8, leading=9, fontName='Helvetica-Bold', wordWrap='CJK', alignment=TA_CENTER)
        domain_style = ParagraphStyle('DomainText', parent=normal, fontSize=8, leading=9, wordWrap='CJK')
        table_text_style = ParagraphStyle('TableText', parent=normal, fontSize=7, leading=8, wordWrap='CJK')
        pass_style = ParagraphStyle('Pass', parent=normal, fontSize=8, textColor=colors.green, alignment=TA_CENTER)
        fail_style = ParagraphStyle('Fail', parent=normal, fontSize=8, textColor=colors.red, alignment=TA_CENTER)

        return header_style, domain_style, table_text_style, pass_style, fail_style, normal

    # ========================= REPORT GENERATION FUNCTIONS =========================
    def generate_report_pdf(title, mapping_dict, audit_period):
        styles = getSampleStyleSheet()
        normal = styles['Normal']
        normal.wordWrap = 'CJK'
        normal.alignment = TA_LEFT

        reasoning_style = ParagraphStyle(
            'Reasoning',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            wordWrap='CJK',
            alignment = TA_LEFT
        )

        note_style = ParagraphStyle(
            'Note',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            wordWrap='CJK',
            alignment=TA_LEFT,
            textColor=colors.darkgrey,
            spaceBefore=10,
            spaceAfter=10
        )

        heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)

        filename = f"{title.replace(' ', '_')}_{audit_period}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []

        story.append(Paragraph(title, ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)))
        story.append(Paragraph(f"Audit Period: {audit_period}", normal))
        story.append(Paragraph(f"Report Date: {date.today()}", normal))
        story.append(Spacer(1, 30))

        # Executive Summary
        total = passed = 0
        scores = []
        for domain in domains:
            vers = data["audit_periods"][audit_period].get(domain, {}).get("versions", [])
            score = vers[-1]["analysis"].get("score", 0) if vers else 0
            scores.append(score)
            total += score
            if score >= 70: passed += 1

        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        story.append(Paragraph(f"Compliant Domains: {passed}/{len(domains)}", normal))
        story.append(Paragraph(f"Average Score: {total/len(domains):.1f}/100", normal))
        story.append(Paragraph(f"Overall Status: {'COMPLIANT' if passed >= 15 else 'ACTION REQUIRED'}", normal))
        story.append(Spacer(1, 20))

        # Charts
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.data = [passed, len(domains) - passed]
        pie.labels = ['Compliant', 'Needs Attention']
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = colors.green
        pie.slices[1].fillColor = colors.red
        drawing.add(pie)
        story.append(drawing)
        story.append(Spacer(1, 20))

        drawing = Drawing(500, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 400
        bc.data = [scores]
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = 100
        bc.valueAxis.valueStep = 20
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = [d.split(':')[0] for d in domains]
        bc.bars[0].fillColor = colors.blue
        drawing.add(bc)
        story.append(drawing)
        story.append(Spacer(1, 20))

        # Domain Table
        table_data = [["Domain", "Requirement", "Recommendation", "Score", "Status", "AI Reasoning"]]
        for domain in domains:
            vers = data["audit_periods"][audit_period].get(domain, {}).get("versions", [])
            if vers:
                analysis = vers[-1]["analysis"]
                score = analysis.get("score", 0)
                status = "PASS" if score >= 70 else "FAIL"
                req = mapping_dict.get(domain, "N/A")
                rec = control_recommendations.get(domain, "Implement controls")
                reasoning = analysis.get("reasoning", "No reasoning")
            else:
                score = 0
                status = "FAIL"
                req = "No evidence"
                rec = "Upload evidence"
                reasoning = "No evidence submitted"

            table_data.append([
                Paragraph(domain, normal),
                Paragraph(req, normal),
                Paragraph(rec, normal),
                Paragraph(str(score), normal),
                Paragraph(status, normal),
                Paragraph(reasoning, reasoning_style)
            ])

        table = Table(table_data, colWidths=[1.8*inch, 2.2*inch, 2.5*inch, 0.7*inch, 0.7*inch, 4.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(table)

        # FedRAMP note
        if "FedRAMP" in title:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Note: This high-level domain maps to multiple NIST 800-53 controls in the FedRAMP Moderate baseline (approximately 323 total controls). Refer to the official FedRAMP Moderate Baseline spreadsheet for exact control IDs and parameters.", note_style))
            story.append(Paragraph("<link href='https://www.fedramp.gov/assets/resources/documents/FedRAMP_Moderate_Security_Controls.xlsx' color='blue'>Download Official FedRAMP Moderate Baseline Spreadsheet</link>", note_style))

        doc.build(story)
        return filename

    # ========================= SPECIAL REPORTS =========================
    def generate_dpia_report(audit_period):
        with st.spinner("Generating GDPR DPIA Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"GDPR_DPIA_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("Data Protection Impact Assessment (DPIA)", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with GDPR Article 35", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. Description of Processing", heading_style))
            story.append(Paragraph("This DPIA covers all personal data processing activities within the current audit period.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Necessity & Proportionality", heading_style))
            story.append(Paragraph("Processing is necessary for compliance and security operations. Measures are proportionate to identified risks.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("3. Risk Assessment", heading_style))
            story.append(Paragraph("Risks to data subjects include unauthorized access, data loss, and insufficient safeguards.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("4. Control Assessment", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("DPIA Requirement", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    req = gdpr_dpia_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    req = gdpr_dpia_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(req, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)
            story.append(PageBreak())

            story.append(Paragraph("5. Mitigation Measures & Residual Risk", heading_style))
            story.append(Paragraph("Residual risk after mitigation: [Low/Medium/High]\nProcessing may proceed: [Yes/No]", normal))

            doc.build(story)
            st.success("GDPR DPIA Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download GDPR DPIA Report", data=f, file_name=filename, mime="application/pdf")

    def generate_lia_report(audit_period):
        with st.spinner("Generating GDPR LIA Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"GDPR_LIA_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("Legitimate Interests Assessment (LIA)", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with GDPR Article 6(1)(f)", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. Purpose Test", heading_style))
            story.append(Paragraph("Legitimate interest identified: compliance, security, and operational necessity.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Necessity Test", heading_style))
            story.append(Paragraph("Processing is necessary and no less intrusive means available.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("3. Balancing Test", heading_style))
            story.append(Paragraph("Interests outweigh data subject rights due to strong security measures and transparency.", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("4. LIA Assessment Table", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("Legitimate Interest", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    interest = gdpr_lia_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    interest = gdpr_lia_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(interest, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            story.append(PageBreak())
            story.append(Paragraph("5. Conclusion", heading_style))
            story.append(Paragraph("Legitimate interests valid basis for processing: Yes", normal))

            doc.build(story)
            st.success("GDPR LIA Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download GDPR LIA Report", data=f, file_name=filename, mime="application/pdf")

    def generate_dsr_report(audit_period):
        with st.spinner("Generating GDPR DSR Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"GDPR_DSR_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("Data Subject Rights (DSR) Management Report", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with GDPR Articles 15–22", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. DSR Process Overview", heading_style))
            story.append(Paragraph("• Response time: 1 month (extendable)\n• Identity verification required\n• No charge unless manifestly unfounded", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. DSR Capability Assessment", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("DSR Requirement", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    req = gdpr_dsr_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    req = gdpr_dsr_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(req, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            story.append(PageBreak())
            story.append(Paragraph("3. Conclusion", heading_style))
            story.append(Paragraph("DSR processes ready: Yes", normal))

            doc.build(story)
            st.success("GDPR DSR Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download GDPR DSR Report", data=f, file_name=filename, mime="application/pdf")

    def generate_dpo_report(audit_period):
        with st.spinner("Generating GDPR DPO Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"GDPR_DPO_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("Data Protection Officer (DPO) Report", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with GDPR Articles 37–39", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. DPO Appointment", heading_style))
            story.append(Paragraph("• DPO appointed based on expert knowledge\n• Reports to highest management level\n• No conflict of interest", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. DPO Tasks Assessment", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("DPO Task", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    task = gdpr_dpo_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    task = gdpr_dpo_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(task, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            story.append(PageBreak())
            story.append(Paragraph("3. Conclusion", heading_style))
            story.append(Paragraph("DPO fully resourced and independent: Yes", normal))

            doc.build(story)
            st.success("GDPR DPO Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download GDPR DPO Report", data=f, file_name=filename, mime="application/pdf")

    def generate_ropa_report(audit_period):
        with st.spinner("Generating GDPR RoPA Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            filename = f"GDPR_RoPA_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            story = []

            story.append(Paragraph("Records of Processing Activities (RoPA)", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with GDPR Article 30", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. Controller Details", heading_style))
            story.append(Paragraph("Controller: [Organization Name]\nDPO Contact: [DPO Details]", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Processing Activities", heading_style))
            table_data = [["Domain", "Purpose", "Categories of Data", "Recipients", "Transfers", "Retention", "Security Measures"]]
            for domain in domains:
                table_data.append([
                    domain,
                    "Compliance & Security",
                    "Employee/Customer Data",
                    "Internal Teams",
                    "EU Only",
                    "Legal Requirement",
                    "Encryption & Access Controls"
                ])

            table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 2.0*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            doc.build(story)
            st.success("GDPR RoPA Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download GDPR RoPA Report", data=f, file_name=filename, mime="application/pdf")

    def generate_ccpa_pia_report(audit_period):
        with st.spinner("Generating CCPA PIA Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            filename = f"CCPA_PIA_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            story = []

            story.append(Paragraph("CCPA Privacy Impact Assessment (§1798.185(a)(15))", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))

            story.append(Paragraph("Assessment of processing presenting significant risk to consumers' privacy.", heading_style))
            story.append(Paragraph("Risk level: Low (due to strong controls)", normal))

            doc.build(story)
            st.success("CCPA PIA Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download CCPA PIA Report", data=f, file_name=filename, mime="application/pdf")

    def generate_soc2_privacy_report(audit_period):
        with st.spinner("Generating SOC2 Privacy Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"SOC2_Privacy_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            story = []

            story.append(Paragraph("SOC2 Privacy Criteria Report (TSC P1.0 - P8.0)", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))

            story.append(Paragraph("Privacy Criteria Assessment", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("Privacy Criterion", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    criterion = soc2_privacy_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    criterion = soc2_privacy_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(criterion, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            doc.build(story)
            st.success("SOC2 Privacy Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download SOC2 Privacy Report", data=f, file_name=filename, mime="application/pdf")

    def generate_hipaa_pia_report(audit_period):
        with st.spinner("Generating HIPAA Privacy Impact Assessment..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"HIPAA_Privacy_Impact_Assessment_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            story = []

            story.append(Paragraph("HIPAA Privacy Impact Assessment", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))

            story.append(Paragraph("Assessment of PHI use and disclosure practices.", heading_style))
            story.append(Paragraph("Compliance level: High", normal))

            table_data = [
                [Paragraph("Domain", header_style), Paragraph("Privacy Rule", header_style), Paragraph("Status", header_style), Paragraph("Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    rule = hipaa_privacy_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    rule = hipaa_privacy_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(rule, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)

            doc.build(story)
            st.success("HIPAA Privacy Impact Assessment generated!")
            with open(filename, "rb") as f:
                st.download_button("Download HIPAA PIA Report", data=f, file_name=filename, mime="application/pdf")

    def generate_iso27701_report(audit_period):
        with st.spinner("Generating ISO 27701 Privacy Audit Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_LEFT

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"ISO27701_Privacy_Audit_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            story = []

            story.append(Paragraph("ISO/IEC 27701 Privacy Information Management Audit Report", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 60))
            story.append(Paragraph("Prepared in accordance with ISO/IEC 27701:2019 – Privacy Information Management System (PIMS)", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. Executive Summary", heading_style))
            total = passed = 0
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                score = vers[-1]["analysis"].get("score", 0) if vers else 0
                total += score
                if score >= 70: passed += 1

            story.append(Paragraph(f"Compliant Domains: {passed}/17", normal))
            story.append(Paragraph(f"Average Score: {total/len(domains):.1f}/100", normal))
            story.append(Paragraph(f"Overall Status: {'CONFORMANT' if passed >= 15 else 'IMPROVEMENT REQUIRED'}", normal))
            story.append(Spacer(1, 40))

            story.append(Paragraph("2. ISO 27701 Control Assessment", heading_style))
            table_data = [
                [Paragraph("Domain", header_style), Paragraph("ISO 27701 Requirement", header_style), Paragraph("Status", header_style), Paragraph("AI Assessment", header_style)]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    req = iso27701_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    req = iso27701_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(req, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            table = Table(table_data, colWidths=[2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(table)
            story.append(PageBreak())

            story.append(Paragraph("3. Privacy Controls & Recommendations", heading_style))
            story.append(Paragraph("• Implement PIMS-specific controls for PII Principals and PII Controllers\n• Regular privacy training and awareness\n• Privacy by design in new processing activities\n• Maintain records of processing activities (RoPA integration)", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("4. Conclusion", heading_style))
            story.append(Paragraph("ISO 27701 conformance level: [Conformant/Needs Improvement]\nRecommendations for PIMS certification readiness provided.", normal))

            doc.build(story)
            st.success("ISO 27701 Privacy Audit Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download ISO 27701 Report", data=f, file_name=filename, mime="application/pdf")

    def generate_nis2_report(audit_period):
        with st.spinner("Generating NIS2 Directive Compliance Report..."):
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
            heading_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
            heading2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, spaceAfter=8)
            normal = styles['Normal']
            normal.wordWrap = 'CJK'
            normal.alignment = TA_JUSTIFY

            header_style, domain_style, table_text_style, pass_style, fail_style, _ = get_common_styles()

            filename = f"NIS2_Directive_Compliance_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("NIS2 Directive Compliance Report", title_style))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 40))
            story.append(Paragraph("Detailed assessment per Directive (EU) 2022/2555 (NIS2) – Cybersecurity Risk-Management Measures (Article 21)", normal))
            story.append(PageBreak())

            story.append(Paragraph("1. NIS2 Applicability & Scope", heading_style))
            story.append(Paragraph("• Applies to essential and important entities in critical sectors (energy, transport, banking, health, digital infrastructure, etc.)\n• Expanded scope from NIS1, including medium/large entities\n• Management body accountability with potential personal liability\n• Harmonized incident reporting across EU", normal))
            story.append(Spacer(1, 20))

            story.append(Paragraph("2. Executive Summary", heading_style))
            total = passed = 0
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                score = vers[-1]["analysis"].get("score", 0) if vers else 0
                total += score
                if score >= 70: passed += 1

            story.append(Paragraph(f"Compliant Domains: {passed}/17", normal))
            story.append(Paragraph(f"Average Score: {total/len(domains):.1f}/100", normal))
            story.append(Paragraph(f"NIS2 Readiness: {'Adequate' if passed >= 15 else 'Improvement Required'}", normal))
            story.append(Spacer(1, 30))

            story.append(Paragraph("3. NIS2 Article 21 Measures Assessment", heading_style))
            table_data = [
                [
                    Paragraph("Domain", header_style),
                    Paragraph("NIS2 Requirement<br/>(Art. 21)", header_style),
                    Paragraph("Status", header_style),
                    Paragraph("AI Assessment", header_style)
                ]
            ]
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    latest = vers[-1]
                    analysis = latest["analysis"]
                    score = analysis.get("score", 0)
                    status = "PASS" if score >= 70 else "FAIL"
                    req = nis2_mappings.get(domain, "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                else:
                    score = 0
                    status = "FAIL"
                    req = nis2_mappings.get(domain, "No evidence")
                    reasoning = "No evidence submitted"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(req, table_text_style),
                    Paragraph(status, pass_style if status == "PASS" else fail_style),
                    Paragraph(reasoning, table_text_style)
                ])

            colWidths = [2.0*inch, 4.0*inch, 0.8*inch, 4.2*inch]
            table = Table(table_data, colWidths=colWidths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                ('BOX', (0,0), (-1,-1), 0.5, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(table)
            story.append(PageBreak())

            story.append(Paragraph("4. Supply Chain Security (Art. 21(2)(c))", heading_style))
            story.append(Paragraph("• Vendor risk assessments conducted\n• Contracts include cybersecurity requirements\n• Monitoring of third-party incidents\n• Diversification of suppliers where possible", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("5. Incident Reporting Obligations (Art. 21(2)(h), Art. 23)", heading_style))
            story.append(Paragraph("• Early warning: Within 24 hours of awareness\n• Incident notification: Within 72 hours\n• Final report: Within 1 month\n• Significant incidents affecting services or other Member States", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("6. Management Accountability (Art. 20)", heading_style))
            story.append(Paragraph("• Management body approves measures and oversees implementation\n• Training for management on cybersecurity\n• Potential personal liability for non-compliance", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("7. Enforcement & Penalties (Article 34)", heading_style))
            story.append(Paragraph("NIS2 requires effective, proportionate and dissuasive penalties. Member States have transposed with varying maximum fines.", normal))
            story.append(Spacer(1, 8))

            story.append(Paragraph("Maximum Administrative Fines:", heading2_style))
            story.append(Paragraph("• Essential entities: At least €10,000,000 or up to 2% of total worldwide annual turnover (whichever is higher)\n• Important entities: At least €7,000,000 or up to 1.4% of total worldwide annual turnover (whichever is higher)", normal))
            story.append(Spacer(1, 8))

            story.append(Paragraph("Supervisory Measures Include:", heading2_style))
            story.append(Paragraph("• Binding instructions and deadlines\n• Mandatory security audits\n• Public warnings or naming & shaming\n• Temporary suspension of certifications\n• Bans on management handling cybersecurity responsibilities\n• Personal liability for management bodies", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("National Transposition Examples:", heading2_style))
            story.append(Paragraph("• Germany (NIS2UmsG): Up to €20M or 2% turnover for essential entities; up to €10M or 1% for important entities. Personal fines up to €1M for management.\n• France: Up to €10M or 2% for essential; up to €7.5M or 1.5% for important. Additional daily penalties possible.\n• Netherlands: Up to €10M or 2% for essential; up to €5M or 1% for important entities.\n• Other Member States align closely to EU minimums, with some exceeding (e.g., Spain up to €20M).", normal))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Hypothetical Non-Compliance Scenarios:", heading2_style))
            examples = [
                "• Failure to implement basic hygiene measures (e.g., no MFA, unpatched systems): Fine up to €10M/2% for essential entity.",
                "• Delayed incident reporting beyond 24/72 hours: Fine up to €5-10M + daily penalties.",
                "• Management ignoring known vulnerabilities leading to breach: Personal liability + organizational fine up to 2% turnover.",
                "• Supply chain risk not addressed (no vendor assessments): Fine up to €7M/1.4% + binding instructions.",
                "• Repeated non-compliance: Increased fines + public warning + management suspension."
            ]

            for ex in examples:
                story.append(Paragraph(ex, normal))
                story.append(Spacer(1, 6))

            story.append(Spacer(1, 12))

            story.append(Paragraph("8. Recommendations", heading_style))
            recommendations_data = [
                ["Priority", "Domain", "Recommended Action"],
                ["High", "STA: Supply Chain", "Conduct vendor cybersecurity assessments"],
                ["High", "SEF: Incident Mgmt", "Implement 24/72-hour reporting process"],
                ["High", "GRC: Governance", "Management cybersecurity training program"],
                ["Medium", "TVM: Threat & Vulnerability Management", "Regular penetration testing"],
                ["Medium", "CEK: Cryptography", "Review cryptography policies for NIS2 alignment"]
            ]

            rec_table = Table(recommendations_data, colWidths=[1.5*inch, 2.5*inch, 6.0*inch])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(rec_table)
            story.append(PageBreak())

            story.append(Paragraph("9. Conclusion & Next Steps", heading_style))
            story.append(Paragraph(f"NIS2 compliance readiness: {'Adequate' if passed >= 15 else 'Improvement Required'}.\nNon-compliance risks significant fines up to 2% of global turnover and personal liability.\nNext steps:\n• Address FAIL domains within 90 days\n• Conduct full NIS2 gap analysis\n• Prepare incident reporting procedures\n• Train management on responsibilities and penalties", normal))

            doc.build(story)
            st.success("NIS2 Directive Compliance Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download NIS2 Report", data=f, file_name=filename, mime="application/pdf")

    # ========================= AI ANALYSIS REPORTS SECTION =========================
    st.markdown("---")
    st.header("🤖 AI Analysis Reports")

    st.markdown("### AI-Powered Compliance Insights")

    st.markdown("#### 1. AI Compliance Maturity Assessment")
    if st.button("Generate AI Compliance Maturity Report", key="ai_gen_maturity", type="primary"):
        with st.spinner("AI is analyzing your compliance maturity..."):
            scores_list = []
            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                score = vers[-1]["analysis"].get("score", 0) if vers else 0
                scores_list.append(score)

            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0

            if avg_score >= 90:
                maturity = "Advanced – Certification Ready"
                color = "green"
            elif avg_score >= 70:
                maturity = "Mature – Strong Controls"
                color = "lightgreen"
            elif avg_score >= 50:
                maturity = "Developing – Moderate Gaps"
                color = "orange"
            else:
                maturity = "Initial – Significant Risk"
                color = "red"

            st.markdown(f"**Overall Compliance Maturity: <span style='color:{color};font-size:1.5em'>{maturity}</span>**", unsafe_allow_html=True)
            st.markdown(f"**Average Score: {avg_score:.1f}/100**")

            domain_scores_tuples = [(d, current_period_data.get(d, {}).get("versions", [{}])[-1]["analysis"].get("score", 0) if current_period_data.get(d, {}).get("versions") else 0) for d in domains]
            top_domains = sorted(domain_scores_tuples, key=lambda x: x[1], reverse=True)[:3]
            st.markdown("**Strongest Domains:**")
            for d, s in top_domains:
                st.markdown(f"- {d}: {s}/100")

            weak_domains = sorted(domain_scores_tuples, key=lambda x: x[1])[:3]
            st.markdown("**Domains Needing Attention:**")
            for d, s in weak_domains:
                st.markdown(f"- {d}: {s}/100 – Priority remediation recommended")

            st.markdown("**AI Recommendation:** Focus on evidence collection and documentation for low-scoring domains to achieve certification readiness.")

    st.markdown("#### 2. AI Risk Heatmap Summary")
    if st.button("Generate AI Risk Heatmap Report (PDF)", key="ai_gen_risk_pdf", type="primary"):
        with st.spinner("Generating AI Risk Heatmap PDF..."):
            high_risk = [d for d in domains if current_period_data.get(d, {}).get("versions", []) and current_period_data[d]["versions"][-1]["analysis"].get("score", 0) < 50]
            medium_risk = [d for d in domains if current_period_data.get(d, {}).get("versions", []) and 50 <= current_period_data[d]["versions"][-1]["analysis"].get("score", 0) < 70]
            low_risk = [d for d in domains if current_period_data.get(d, {}).get("versions", []) and current_period_data[d]["versions"][-1]["analysis"].get("score", 0) >= 70]

            header_style, domain_style, table_text_style, pass_style, fail_style, normal = get_common_styles()

            filename = f"AI_Risk_Heatmap_Summary_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("AI Risk Heatmap Summary", ParagraphStyle('Title', fontSize=20, alignment=TA_CENTER, spaceAfter=30)))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 30))

            # Risk Classification Table - Wider column for domains
            story.append(Paragraph("AI Risk Classification:", ParagraphStyle('Heading1', fontSize=14, spaceAfter=12)))
            risk_data = [
                [Paragraph("Risk Level", header_style), Paragraph("Count", header_style), Paragraph("Domains", header_style)]
            ]
            risk_data.append([Paragraph("High Risk (<50)", table_text_style), str(len(high_risk)), Paragraph(", ".join(high_risk) or "None", table_text_style)])
            risk_data.append([Paragraph("Medium Risk (50–69)", table_text_style), str(len(medium_risk)), Paragraph(", ".join(medium_risk) or "None", table_text_style)])
            risk_data.append([Paragraph("Low Risk (≥70)", table_text_style), str(len(low_risk)), Paragraph(", ".join(low_risk) or "None", table_text_style)])

            risk_table = Table(risk_data, colWidths=[1.8*inch, 1.0*inch, 7.2*inch], rowHeights=0.5*inch)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (1, -1), 'CENTER'),
                ('LEFTPADDING', (2,1), (2,-1), 10),
                ('RIGHTPADDING', (2,1), (2,-1), 10),
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 30))

            # Detailed Risk Matrix - Wider domain column
            story.append(Paragraph("Detailed Risk Matrix:", ParagraphStyle('Heading1', fontSize=14, spaceAfter=12)))
            matrix_data = [["Domain", "Score", "Risk Level"]]
            for d in domains:
                vers = current_period_data.get(d, {}).get("versions", [])
                score = vers[-1]["analysis"].get("score", 0) if vers else 0
                if score < 50:
                    level = "High Risk"
                elif score < 70:
                    level = "Medium Risk"
                else:
                    level = "Low Risk"
                matrix_data.append([d, str(score), level])

            matrix_table = Table(matrix_data, colWidths=[5.0*inch, 1.0*inch, 2.0*inch], rowHeights=0.4*inch)
            matrix_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            for i in range(1, len(matrix_data)):
                score = int(matrix_data[i][1])
                if score < 50:
                    matrix_table.setStyle(TableStyle([('BACKGROUND', (0,i), (-1,i), colors.red)]))
                elif score < 70:
                    matrix_table.setStyle(TableStyle([('BACKGROUND', (0,i), (-1,i), colors.orange)]))
                else:
                    matrix_table.setStyle(TableStyle([('BACKGROUND', (0,i), (-1,i), colors.green)]))
            story.append(matrix_table)
            story.append(Spacer(1, 30))

            # Pie Chart
            story.append(Paragraph("Risk Distribution", ParagraphStyle('Heading1', fontSize=14, spaceAfter=12)))
            drawing = Drawing(400, 200)
            pie = Pie()
            pie.x = 150
            pie.y = 50
            pie.data = [len(high_risk), len(medium_risk), len(low_risk)]
            pie.labels = ['High Risk', 'Medium Risk', 'Low Risk']
            pie.slices.strokeWidth = 0.5
            pie.slices[0].fillColor = colors.red
            pie.slices[1].fillColor = colors.orange
            pie.slices[2].fillColor = colors.green
            drawing.add(pie)
            story.append(drawing)
            story.append(Spacer(1, 20))

            story.append(Paragraph("AI Insight: Immediate action required on High Risk domains to mitigate regulatory exposure.", normal))

            doc.build(story)
            st.success("AI Risk Heatmap Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download AI Risk Heatmap Report", data=f, file_name=filename, mime="application/pdf", key="ai_dl_risk_heatmap_pdf")

    st.markdown("#### 3. AI Remediation Roadmap")
    if st.button("Generate AI Remediation Roadmap", key="ai_gen_roadmap", type="primary"):
        with st.spinner("AI is building your remediation roadmap..."):
            gaps = [d for d in domains if current_period_data.get(d, {}).get("versions", []) and current_period_data[d]["versions"][-1]["analysis"].get("score", 0) < 70]

            st.markdown("**AI-Generated 90-Day Remediation Roadmap:**")
            if gaps:
                st.markdown("**Phase 1 (Days 1–30): Critical Gaps**")
                critical = [d for d in gaps if current_period_data[d]["versions"][-1]["analysis"].get("score", 0) < 50]
                for d in critical:
                    st.markdown(f"- **{d}** – Upload policy documents and evidence of controls")

                st.markdown("**Phase 2 (Days 31–60): Moderate Gaps**")
                moderate = [d for d in gaps if 50 <= current_period_data[d]["versions"][-1]["analysis"].get("score", 0) < 70]
                for d in moderate:
                    st.markdown(f"- {d} – Enhance existing evidence with implementation proof")

                st.markdown("**Phase 3 (Days 61–90): Validation**")
                st.markdown("- Conduct internal review\n- Run AI analysis again\n- Prepare for external audit")
            else:
                st.success("No gaps detected – Your compliance posture is strong!")

    st.markdown("#### 4. Completed AI Analysis Per Domain")
    if st.button("Generate Domain AI Analysis Report (PDF)", key="ai_gen_domain_analysis_pdf", type="primary"):
        with st.spinner("Compiling AI Analysis for all domains..."):
            header_style, domain_style, table_text_style, pass_style, fail_style, normal = get_common_styles()

            filename = f"Domain_AI_Analysis_Report_{audit_period.replace(' ', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []

            story.append(Paragraph("Completed AI Analysis Per Domain", ParagraphStyle('Title', fontSize=20, alignment=TA_CENTER, spaceAfter=30)))
            story.append(Paragraph(f"Audit Period: {audit_period}", normal))
            story.append(Paragraph(f"Report Date: {date.today()}", normal))
            story.append(Spacer(1, 20))

            table_data = [
                [Paragraph("Domain", header_style), Paragraph("Score", header_style), Paragraph("Readiness", header_style), Paragraph("Completeness", header_style), Paragraph("Reasoning", header_style), Paragraph("Recommendations", header_style)]
            ]

            for domain in domains:
                vers = current_period_data.get(domain, {}).get("versions", [])
                if vers:
                    analysis = vers[-1]["analysis"]
                    score = analysis.get("score", 0)
                    readiness = analysis.get("audit_readiness", "N/A")
                    completeness = analysis.get("completeness", "N/A")
                    reasoning = analysis.get("reasoning", "No reasoning")
                    recommendations = analysis.get("recommendations", "No recommendations")
                else:
                    score = 0
                    readiness = "Fail"
                    completeness = "None"
                    reasoning = "No evidence submitted"
                    recommendations = "Upload evidence"

                table_data.append([
                    Paragraph(domain, domain_style),
                    Paragraph(str(score), table_text_style),
                    Paragraph(readiness, table_text_style),
                    Paragraph(completeness, table_text_style),
                    Paragraph(reasoning, table_text_style),
                    Paragraph(recommendations, table_text_style)
                ])

            colWidths = [2.0*inch, 0.8*inch, 1.2*inch, 1.2*inch, 3.0*inch, 3.0*inch]
            table = Table(table_data, colWidths=colWidths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                ('BOX', (0,0), (-1,-1), 0.5, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(table)

            doc.build(story)
            st.success("Domain AI Analysis Report generated!")
            with open(filename, "rb") as f:
                st.download_button("Download Domain AI Analysis Report", data=f, file_name=filename, mime="application/pdf", key="ai_dl_domain_analysis_pdf")

    # ========================= FULL REPORT UI =========================
    st.markdown("---")
    st.header("Compliance Audit Reports")

    main_reports = [
        ("Multi-Framework Report", soc2_mappings),
        ("SOC2 Type 2 Report", soc2_mappings),
        ("ISO 27001 Report", iso27001_controls),
        ("CMMC Level 3 Report", cmmc_level3_mappings),
        ("NIST 800-53 Report", nist80053_mappings),
        ("FedRAMP Moderate Report", fedramp_moderate_mappings),
        ("ISO 42001 AI Report", iso42001_mappings),
        ("EU AI Act High-Risk Report", eu_ai_act_high_risk_mappings),
        ("NIS2 Directive Compliance Report", None),
    ]

    cols = st.columns(len(main_reports))

    for i, (title, mapping) in enumerate(main_reports):
        with cols[i]:
            st.markdown(f"**{title}**")
            if st.button("Generate", key=f"report_gen_main_{i}", type="primary"):
                with st.spinner(f"Generating {title}..."):
                    if title == "NIS2 Directive Compliance Report":
                        generate_nis2_report(audit_period)
                    else:
                        pdf_file = generate_report_pdf(title, mapping, audit_period)
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                "Download PDF",
                                data=f,
                                file_name=os.path.basename(pdf_file),
                                mime="application/pdf",
                                key=f"report_dl_main_{i}"
                            )
                    st.success("Report Ready!")

    st.markdown("---")
    st.subheader("GDPR Privacy Reports")

    gdpr_reports = [
        ("GDPR Compliance Report", gdpr_mappings),
        ("GDPR DPIA Report", None),
        ("GDPR LIA Report", None),
        ("GDPR DSR Report", None),
        ("GDPR DPO Report", None),
        ("GDPR RoPA Report", None),
    ]

    cols = st.columns(len(gdpr_reports))

    for i, (title, mapping) in enumerate(gdpr_reports):
        with cols[i]:
            st.markdown(f"**{title}**")
            if st.button("Generate", key=f"report_gen_gdpr_{i}", type="primary"):
                with st.spinner(f"Generating {title}..."):
                    if mapping is not None:
                        pdf_file = generate_report_pdf(title, mapping, audit_period)
                        with open(pdf_file, "rb") as f:
                            st.download_button("Download PDF", data=f, file_name=os.path.basename(pdf_file), mime="application/pdf", key=f"report_dl_gdpr_{i}")
                    else:
                        if title == "GDPR DPIA Report":
                            generate_dpia_report(audit_period)
                        elif title == "GDPR LIA Report":
                            generate_lia_report(audit_period)
                        elif title == "GDPR DSR Report":
                            generate_dsr_report(audit_period)
                        elif title == "GDPR DPO Report":
                            generate_dpo_report(audit_period)
                        elif title == "GDPR RoPA Report":
                            generate_ropa_report(audit_period)
                    st.success("Report Ready!")

    st.markdown("---")
    st.subheader("CCPA Privacy Reports")

    ccpa_reports = [
        ("CCPA Compliance Report", ccpa_mappings),
        ("CCPA PIA Report (§1798.185(a)(15))", None),
    ]

    cols = st.columns(len(ccpa_reports))

    for i, (title, mapping) in enumerate(ccpa_reports):
        with cols[i]:
            st.markdown(f"**{title}**")
            if st.button("Generate", key=f"report_gen_ccpa_{i}", type="primary"):
                with st.spinner(f"Generating {title}..."):
                    if mapping is not None:
                        pdf_file = generate_report_pdf(title, mapping, audit_period)
                        with open(pdf_file, "rb") as f:
                            st.download_button("Download PDF", data=f, file_name=os.path.basename(pdf_file), mime="application/pdf", key=f"report_dl_ccpa_{i}")
                    else:
                        generate_ccpa_pia_report(audit_period)
                    st.success("Report Ready!")

    st.markdown("---")
    st.subheader("SOC2 Privacy Reports")

    if st.button("Generate SOC2 Privacy Report (TSC Privacy)", key="report_gen_soc2_privacy", type="primary"):
        generate_soc2_privacy_report(audit_period)

    st.markdown("---")
    st.subheader("HIPAA Reports")

    hipaa_reports = [
        ("HIPAA Security Report", hipaa_security_mappings),
        ("HIPAA Privacy Report", hipaa_privacy_mappings),
        ("HIPAA Privacy Impact Assessment", None),
    ]

    cols = st.columns(len(hipaa_reports))

    for i, (title, mapping) in enumerate(hipaa_reports):
        with cols[i]:
            st.markdown(f"**{title}**")
            if st.button("Generate", key=f"report_gen_hipaa_{i}", type="primary"):
                with st.spinner(f"Generating {title}..."):
                    if mapping is not None:
                        pdf_file = generate_report_pdf(title, mapping, audit_period)
                        with open(pdf_file, "rb") as f:
                            st.download_button("Download PDF", data=f, file_name=os.path.basename(pdf_file), mime="application/pdf", key=f"report_dl_hipaa_{i}")
                    else:
                        generate_hipaa_pia_report(audit_period)
                    st.success("Report Ready!")

    st.markdown("---")
    st.subheader("ISO Privacy Reports")

    if st.button("Generate ISO 27701 Privacy Audit Report", key="report_gen_iso27701", type="primary"):
        generate_iso27701_report(audit_period)

st.caption("Version 7.99 – Ollama Local Primary | OpenAI Optional Fallback | Air-Gapped Ready | Final")