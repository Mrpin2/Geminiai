import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import os
import tempfile
import io
import time # For animations/sleep

# Try to import google.generativeai, show error if not found
try:
    # Use the new 'google-genai' library for the latest SDK
    from google import generativeai as genai
except ImportError:
    st.error("The 'google-genai' library is not installed. Please install it by running: `pip install google-genai`")
    st.stop()

# --- Pydantic Models (Enhanced) ---
class LineItem(BaseModel):
    description: str = Field(..., description="Description of the item or service.")
    quantity: float = Field(..., description="Quantity of the item, e.g., 1.0, 5.5.")
    gross_worth: float = Field(..., description="Gross value of the line item before taxes.")
    hsn_sac_code: Optional[str] = Field(None, description="HSN (Harmonized System of Nomenclature) or SAC (Services Accounting Code) if available.")

class Invoice(BaseModel):
    invoice_number: str = Field(..., description="Unique invoice identification number.")
    date: str = Field(..., description="Date of the invoice in DD/MM/YYYY format.")
    seller_name: str = Field(..., description="Name of the seller/supplier.")
    seller_gstin: str = Field(..., description="GSTIN (Goods and Services Tax Identification Number) of the seller.")
    buyer_name: str = Field(..., description="Name of the buyer/recipient.")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer, if available.")
    place_of_supply: Optional[str] = Field(None, description="Place of Supply for GST purposes.")
    line_items: List[LineItem] = Field(..., description="List of individual line items in the invoice.")
    total_gross_worth: float = Field(..., description="Total gross value of all line items before taxes.")
    cgst_amount: Optional[float] = Field(None, description="Central Goods and Services Tax amount.")
    sgst_amount: Optional[float] = Field(None, description="State Goods and Services Tax amount.")
    igst_amount: Optional[float] = Field(None, description="Integrated Goods and Services Tax amount.")
    total_tax_amount: Optional[float] = Field(None, description="Total tax amount (sum of CGST, SGST, IGST).")
    total_payable_amount: float = Field(..., description="Total amount payable including taxes.")
    expense_ledger_suggestion: Optional[str] = Field(None, description="Suggested accounting ledger for the expense (e.g., 'Office Supplies', 'Professional Fees').")
    tds_applicability: Optional[str] = Field(None, description="TDS (Tax Deducted at Source) applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain').")
    rcm_applicability: Optional[str] = Field(None, description="Reverse Charge Mechanism (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain').")
    currency: str = Field("INR", description="Currency of the invoice amount, defaults to INR.")

# --- Gemini API Interaction Function ---
def extract_structured_data(
    gemini_model_id: str, # No longer needs client_instance directly
    file_path: str,
    pydantic_schema: BaseModel,
    progress_callback=None
):
    display_name = os.path.basename(file_path)
    gemini_file_resource = None

    try:
        if progress_callback:
            progress_callback(0.1, f"Uploading '{display_name}' to Gemini File API...")
        
        # Use genai.upload_file directly
        gemini_file_resource = genai.upload_file(
            path=file_path,
            display_name=display_name.split('.')[0]
        )
        if progress_callback:
            progress_callback(0.4, f"'{display_name}' uploaded. Gemini file name: {gemini_file_resource.name}")

        prompt = (
            "Extract all clear and relevant information from the invoice, specifically adhering to Indian tax "
            "standards for dates (DD/MM/YYYY or DD-MM-YYYY), GSTINs, and HSN/SAC codes. "
            "Identify the invoice number, date, seller and buyer names, their GSTINs, "
            "place of supply, and detailed line items including description, quantity, gross worth, and HSN/SAC codes. "
            "Calculate and explicitly state CGST, SGST, IGST, total tax amount, and the **total payable amount**. "
            "Classify the nature of expense and suggest an applicable ledger type "
            "(e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription', 'Consulting Charges'). "
            "Determine TDS applicability (e.g., 'Yes - Section 194C', 'Yes - Section 194J', 'No', 'Uncertain'). "
            "Determine Reverse Charge Mechanism (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain'). "
            "Handle missing data by setting Optional fields to null/None. "
            "If a value is clearly zero, represent it as 0.0 for floats. "
            "For dates, prefer DD/MM/YYYY. Ensure all financial figures are extracted precisely. "
            "The currency should be 'INR' by default unless explicitly stated otherwise."
        )

        if progress_callback:
            progress_callback(0.6, f"Sending '{display_name}' to Gemini model '{gemini_model_id}' for extraction...")
        
        # Load the model directly
        model = genai.GenerativeModel(gemini_model_id)

        response = model.generate_content(
            contents=[prompt, gemini_file_resource],
            generation_config={'response_mime_type': 'application/json', 'response_schema': pydantic_schema.model_json_schema()}
        )
        
        if progress_callback:
            progress_callback(0.9, f"Data extracted for '{display_name}'.")

        # Use .text and then parse if .parsed is not directly available or causes issues
        import json
        return pydantic_schema.model_validate_json(response.text)

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        import traceback
        st.exception(e) # Display full traceback
        return None
    finally:
        if gemini_file_resource:
            try:
                if progress_callback:
                    progress_callback(1.0, f"Cleaning up '{gemini_file_resource.name}' from Gemini File API...")
                genai.delete_file(name=gemini_file_resource.name) # Use genai.delete_file
            except Exception as e_del:
                st.warning(f"Could not delete '{gemini_file_resource.name}' from Gemini File API: {e_del}. Manual cleanup may be required.")


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AI-Powered Invoice Extractor", page_icon="🤖")

# Custom header with emoji and brand
st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #2e86de !important; /* Made color !important for header */
        text-align: center;
        margin-bottom: 1em;
        animation: fadeIn 2s;
    }
    .stButton>button {
        background-color: #2e86de;
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1a6cb2;
        transform: translateY(-2px);
    }
    .stProgress > div > div > div > div {
        background-color: #2e86de !important;
    }
    /* Targeted CSS for the st.info box text */
    .stAlert {
        background-color: #e0f2f7; /* A slightly darker blue for better contrast as per previous iteration */
        color: black !important; /* Ensures the info box's primary text color is black */
    }
    .stAlert div[data-testid="stMarkdownContainer"] p,
    .stAlert div[data-testid="stMarkdownContainer"] li {
        color: black !important; /* Force black for paragraphs and list items inside the markdown container within the alert */
        font-weight: bold !important; /* Make text bold */
    }
    .stAlert div[data-testid="stMarkdownContainer"] b {
        color: black !important; /* Ensure bold tags also remain black */
        font-weight: bolder !important; /* Even bolder */
    }

    @keyframes fadeIn {
      0% { opacity: 0; }
      100% { opacity: 1; }
    }
    </style>
    <h1 class="main-header">📄 AI Invoice Assistant 🚀</h1>
    """, unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")

# --- API Key Handling ---
# 1. Try to get API key from st.secrets first
default_api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")

# 2. Allow user to input/override in sidebar. Pre-fill with secret if available.
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    value=default_api_key_from_secrets
)

# If the user has entered a key in the input box, use that. Otherwise, stick with the one from secrets.
# This prioritizes user input if they type something, otherwise relies on secrets.
effective_gemini_api_key = gemini_api_key if gemini_api_key else default_api_key_from_secrets

# --- Configure Gemini API as early as possible ---
if effective_gemini_api_key:
    try:
        genai.configure(api_key=effective_gemini_api_key)
        # st.sidebar.success("✅ Gemini API configured.") # No need to show this success, sidebar success is for admin
    except Exception as e:
        st.sidebar.error(f"❌ Failed to configure Gemini API: {e}")
        # Clear the key if it failed to configure to prevent re-attempts with bad key
        effective_gemini_api_key = ""

# Define the password for processing
# Access it from secrets.toml. If not found, fall back to "Rajeev".
ACCESS_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "Rajeev") 

# Changed label here
user_entered_password = st.sidebar.text_input("Enter Password for Admin Panel:", type="password")

# Display admin panel activated message
if user_entered_password == ACCESS_PASSWORD and user_entered_password != "":
    st.sidebar.success("🎉 Admin Panel Activated!")
    # Optionally, you could set a session state variable here
    # st.session_state.admin_activated = True

DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
gemini_model_id_input = st.sidebar.text_input(
    "Gemini Model ID for Extraction:",
    DEFAULT_GEMINI_MODEL_ID
)
st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure the model supports schema-based JSON output.")

# Instruction text using standard markdown within st.info
st.info(
    "**Instructions:**\n"
    "1. Enter your **Gemini API Key** and **Password** in the sidebar.\n"
    "2. Upload one or more **PDF invoice files**.\n"
    "3. Click **'🚀 Process Invoices'** to extract data.\n"
    "The extracted data will be displayed in a table and available for download as Excel."
)


uploaded_files = st.file_uploader(
    "📂 Choose PDF Invoice Files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF files containing invoices for data extraction."
)

# Initialize session state for results and client
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
# No longer need 'client' in session state with the new SDK clientless setup for common ops
# if 'client' not in st.session_state:
#     st.session_state.client = None

col1, col2 = st.columns([0.6, 0.4])

with col1:
    process_button = st.button("🚀 Process Invoices", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear Results", use_container_width=True)

if clear_button:
    st.session_state.summary_rows = []
    # st.session_state.client = None # Not needed
    uploaded_files = [] # This won't clear the file uploader directly on refresh, but clears logic
    st.experimental_rerun() # Rerun to clear uploader and display

if process_button:
    # --- Corrected API Key Check ---
    if not effective_gemini_api_key: # Check the effective_gemini_api_key
        st.error("❗ Please enter your Gemini API Key in the sidebar.")
    elif user_entered_password != ACCESS_PASSWORD: # Compare against the defined ACCESS_PASSWORD
        st.error("🔒 Incorrect password. Please enter the correct password to proceed.")
    elif not uploaded_files:
        st.error("⬆️ Please upload at least one PDF file to process.")
    elif not gemini_model_id_input:
        st.error("💡 Please specify a Gemini Model ID in the sidebar.")
    else:
        # If we reach here, effective_gemini_api_key should already be configured
        # The genai.configure() happens earlier for consistency.
        st.session_state.summary_rows = []
        status_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file_obj in enumerate(uploaded_files):
            st.markdown("---")
            status_text.info(f"⏳ Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files})")
            temp_file_path = None
            
            def update_progress(percentage, message):
                progress_bar.progress(int(percentage * 100))
                status_text.text(f"Processing {uploaded_file_obj.name}: {message}")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file_obj.getvalue())
                    temp_file_path = tmp.name

                extracted_data = extract_structured_data(
                    gemini_model_id=gemini_model_id_input,
                    file_path=temp_file_path,
                    pydantic_schema=Invoice,
                    progress_callback=update_progress
                )

                if extracted_data:
                    st.success(f"🎉 Successfully extracted data from **{uploaded_file_obj.name}**")
                    
                    # Handle potential None values for display
                    cgst = extracted_data.cgst_amount if extracted_data.cgst_amount is not None else 0.0
                    sgst = extracted_data.sgst_amount if extracted_data.sgst_amount is not None else 0.0
                    igst = extracted_data.igst_amount if extracted_data.igst_amount is not None else 0.0
                    total_tax = extracted_data.total_tax_amount if extracted_data.total_tax_amount is not None else (cgst + sgst + igst) # Fallback calculation
                    pos = extracted_data.place_of_supply if extracted_data.place_of_supply else "N/A"
                    buyer_gstin_display = extracted_data.buyer_gstin or "N/A"
                    expense_ledger = extracted_data.expense_ledger_suggestion or "Uncategorized"
                    tds_status = extracted_data.tds_applicability or "Not Specified"
                    rcm_status = extracted_data.rcm_applicability or "Not Specified"

                    # Create a more detailed narration including line items if possible
                    line_item_summary = "; ".join([
                        f"{item.description} (Qty: {item.quantity}, Worth: {item.gross_worth:.2f})"
                        for item in extracted_data.line_items
                    ])
                    if not line_item_summary:
                        line_item_summary = "No detailed line items extracted."

                    narration = (
                        f"Invoice **{extracted_data.invoice_number}** dated **{extracted_data.date}** "
                        f"from **{extracted_data.seller_name}** (GSTIN: {extracted_data.seller_gstin}) "
                        f"to **{extracted_data.buyer_name}** (GSTIN: {buyer_gstin_display}). "
                        f"Total Gross Worth: **₹{extracted_data.total_gross_worth:.2f}**. "
                        f"Taxes: CGST ₹{cgst:.2f}, SGST ₹{sgst:.2f}, IGST ₹{igst:.2f}. Total Tax: ₹{total_tax:.2f}. "
                        f"**Total Payable: ₹{extracted_data.total_payable_amount:.2f}**. "
                        f"Place of Supply: {pos}. Suggested Expense Ledger: **{expense_ledger}**. "
                        f"TDS: {tds_status}. RCM: {rcm_status}. "
                        f"Details: {line_item_summary}"
                    )
                    
                    st.session_state.summary_rows.append({
                        "File Name": uploaded_file_obj.name,
                        "Invoice Number": extracted_data.invoice_number,
                        "Date": extracted_data.date,
                        "Seller Name": extracted_data.seller_name,
                        "Seller GSTIN": extracted_data.seller_gstin,
                        "Buyer Name": extracted_data.buyer_name,
                        "Buyer GSTIN": buyer_gstin_display,
                        "Total Gross Worth (₹)": f"{extracted_data.total_gross_worth:.2f}",
                        "CGST (₹)": f"{cgst:.2f}",
                        "SGST (₹)": f"{sgst:.2f}",
                        "IGST (₹)": f"{igst:.2f}",
                        "Total Tax (₹)": f"{total_tax:.2f}",
                        "Total Payable (₹)": f"{extracted_data.total_payable_amount:.2f}",
                        "Place of Supply": pos,
                        "Expense Ledger": expense_ledger,
                        "TDS Applicability": tds_status,
                        "RCM Applicability": rcm_status,
                        "Narration": narration,
                        "Line Items": ", ".join([f"{item.description} (Qty: {item.quantity}, Gross: {item.gross_worth})" for item in extracted_data.line_items])
                    })
                else:
                    st.warning(f"⚠️ Failed to extract data or no data returned for **{uploaded_file_obj.name}**.")

            except Exception as e_outer:
                st.error(f"❌ An unexpected error occurred while processing **{uploaded_file_obj.name}**: {e_outer}")
                st.exception(e_outer)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    # No need for a separate message, progress bar takes care of it
            progress_bar.progress((i + 1) / total_files)
            time.sleep(0.1) # Small delay for animation effect

        st.markdown("---")
        if st.session_state.summary_rows:
            st.balloons()
            status_text.success("✅ All invoices processed successfully!")
        else:
            status_text.warning("No data extracted from any of the uploaded invoices.")

if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    
    # Format financial columns for better display
    financial_cols = ["Total Gross Worth (₹)", "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Tax (₹)", "Total Payable (₹)"]
    for col in financial_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"₹{float(x):,.2f}" if pd.notnull(x) else "N/A")

    st.dataframe(df, use_container_width=True)

    # Provide download link for Excel
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Convert financial columns back to numeric for Excel if needed, or keep as formatted strings
        # For Excel, it's often better to save as raw numbers for calculations
        df_for_excel = pd.DataFrame(st.session_state.summary_rows)
        for col in financial_cols:
            if col in df_for_excel.columns:
                df_for_excel[col] = pd.to_numeric(df_for_excel[col], errors='coerce') # Convert back to numbers

        df_for_excel.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the extracted invoice data as an Excel spreadsheet."
    )
elif not uploaded_files and not process_button: # Only show this if nothing has been uploaded or processed yet
    st.info("Upload PDF files and click '🚀 Process Invoices' to see results.")

st.markdown("---")
st.caption("Developed using Gemini AI and Streamlit.")
