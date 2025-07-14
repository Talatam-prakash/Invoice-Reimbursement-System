
import streamlit as st
import requests

# Configuration
API_BASE = "http://localhost:8000"  

st.set_page_config(page_title="Invoice Reimbursement System", layout="wide")

st.title("Invoice Reimbursement System (FastAPI + Streamlit)")

# --- TAB: Analyze Invoices ---
tab1, tab2 = st.tabs(["Analyze Invoices", "Query Invoices"])

with tab1:
    st.header("Upload HR Policy & Invoice ZIP")

    policy_file = st.file_uploader("Upload HR Policy (PDF or DOCX)", type=["pdf", "docx"])
    invoice_zip = st.file_uploader("Upload ZIP with Invoice PDFs", type=["zip"])
    employee_name = st.text_input("Default Employee Name (optional)", value="Unknown")

    if st.button("Analyze Invoices"):
        if not policy_file or not invoice_zip:
            st.warning("Please upload both HR Policy and Invoice ZIP file.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    files = {
                        "policy_file": (policy_file.name, policy_file, policy_file.type),
                        "invoice_zip": (invoice_zip.name, invoice_zip, invoice_zip.type)
                    }
                    data = {
                        "employee_name": employee_name
                    }
                    response = requests.post(f"{API_BASE}/analyze-invoices/", files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(" Analysis Complete!")
                        st.json(result)
                    else:
                        st.error(f" Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f" Exception: {e}")

# --- TAB: Query Invoices ---
with tab2:
    st.header("Ask Questions About Processed Invoices")

    user_query = st.text_input("Enter your query", placeholder="e.g. Show declined invoices for John in June")
    status_filter = st.text_input("Filter by status (optional)", placeholder="Fully Reimbursed, Partially Reimbursed, Declined")
    emp_filter = st.text_input("Filter by employee name (optional)")

    if st.button("🤖 Ask the Bot"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            params = {
                "query": user_query
            }
            if status_filter:
                params["status"] = status_filter
            if emp_filter:
                params["employee_name"] = emp_filter

            with st.spinner("Querying RAG chatbot..."):
                try:
                    response = requests.get(f"{API_BASE}/query-invoices/", params=params)
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Answer:")
                        st.markdown(data["response"])
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Exception: {e}")
