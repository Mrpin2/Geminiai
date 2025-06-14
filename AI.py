import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import os
import tempfile
import io
import time # For animations/sleep
import copy # For deep copying schema

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

# --- Helper Function to convert Pydantic Schema to Gemini API Compatible Schema ---
def pydantic_schema_to_gemini_schema(pydantic_model_schema: dict) -> dict:
    """
    Converts a Pydantic V2 JSON schema (with $defs) into a format
    compatible with google.generativeai.types.Schema by inlining definitions.
    This function recursively processes the schema to resolve references.
    """
    # Make a deep copy to avoid modifying the original Pydantic schema
    schema_copy = copy.deepcopy(pydantic_model_schema)
    
    # Extract $defs for easy lookup and then remove them from the main schema copy
    definitions = schema_copy.pop("$defs", {})

    def resolve_refs_recursive(sub_schema: dict) -> dict:
        """
        Recursively resolves '$ref' in a sub-schema by inlining definitions.
        This function handles both direct '$ref' and '$ref' within 'items' for arrays.
        """
        if "$ref" in sub_schema:
            ref_path = sub_schema["$ref"].split("/")
            if len(ref_path) == 3 and ref_path[1] == "$defs":
                ref_name = ref_path[2]
                if ref_name in definitions:
                    # Recursively resolve the definition itself before returning
                    return resolve_refs_recursive(definitions[ref_name])
            return sub_schema # If $ref is not a local $defs or not found, return as is

        if sub_schema.get("type") == "object" and "properties" in sub_schema:
            # Recursively process properties of an object
            for prop_name, prop_details in sub_schema["properties"].items():
                sub_schema["properties"][prop_name] = resolve_refs_recursive(prop_details)
        
        if sub_schema.get("type") == "array" and "items" in sub_schema:
            # Recursively process items of an array
            sub_schema["items"] = resolve_refs_recursive(sub_schema["items"])
            
        return sub_schema

    # Start resolving from the root of the schema (after popping $defs)
    # Remove top-level metadata fields that `genai.types.Schema` doesn't expect directly
    # e.g., 'title', '$schema', 'description'
    filtered_root_schema = {
        k: v for k, v in schema_copy.items() 
        if k not in ["title", "$schema", "description"] 
    }
    
    return resolve_refs_recursive(filtered_root_schema)


# --- Gemini API Interaction Function ---
def extract_structured_data(
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
    progress_callback=None
):
    """
    Extracts structured data from a PDF invoice using the Gemini API.

    Args:
        gemini_model_id (str): The ID of the Gemini model to use (e.g., 'gemini-1.5-flash-latest').
        file_path (str): The local path to the PDF invoice file.
        pydantic_schema (BaseModel): The Pydantic schema to enforce for the extracted data.
        progress_callback (callable, optional): A function to update progress (percentage, message).

    Returns:
        Invoice: An Invoice Pydantic model instance with the extracted data, or None if an error occurs.
    """
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

        # Generate the Pydantic JSON schema and then convert it
        raw_pydantic_json_schema = pydantic_schema.model_json_schema()
        # Convert the Pydantic schema to the Gemini-compatible schema
        gemini_compatible_schema = pydantic_schema_to_gemini_schema(raw_pydantic_json_schema)

        response = model.generate_content(
            contents=[prompt, gemini_file_resource],
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': gemini_compatible_schema # Use the converted schema here
            }
        )
        
        if progress_callback:
            progress_callback(0.9, f"Data extracted for '{display_name}'.")

        # Use .text to get the raw JSON string and then parse it using Pydantic's model_validate_json
        return pydantic_schema.model_validate_json(response.text)

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        import traceback
        st.exception(e) # Display full traceback for debugging
        return None
    finally:
        # Clean up the uploaded file from Gemini File API
        if gemini_file_resource:
            try:
                if progress_callback:
                    progress_callback(1.0, f"Cleaning up '{gemini_file_resource.name}' from Gemini File API...")
                genai.delete_file(name=gemini_file_resource.name) # Use genai.delete_file
            except Exception as e_del:
                st.warning(f"Could not delete '{gemini_file_resource.name}' from Gemini File API: {e_del}. Manual cleanup may be required.")


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AI-Powered Invoice Extractor", page_icon="🤖")

# Custom header with emoji and brand and general CSS
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
    /* Reverted st.info styling for better readability */
    .stAlert {
        background-color: #e0f2f7; /* A light blue */
        color: #333333; /* Darker text for contrast */
    }
    .stAlert div[data-testid="stMarkdownContainer"] p,
    .stAlert div[data-testid="stMarkdownContainer"] li,
    .stAlert div[data-testid="stMarkdownContainer"] b {
        color: #333333 !important; /* Ensure text inside info box is readable */
        font-weight: normal; /* Remove bold from info text to be less aggressive */
    }

    /* CSS for uploaded file names */
    div[data-testid="stFileUploader"] span.css-10trblm.e16nr0p30,
    div[data-testid="stFileUploader"] div.uploadedFileName {
        color: black !important;
    }
    div[data-testid="stFileUploader"] button[aria-label="Remove file"] {
        color: black !important;
    }

    @keyframes fadeIn {
      0% { opacity: 0; }
      100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True) # CSS block ends here

# --- Main Header - Now using inline style for robustness ---
st.markdown(
    """
    <h1 style="font-size: 2.5em; font-weight: bold; color: #2e86de !important; text-align: center; margin-bottom: 1em; animation: fadeIn 2s;">
        📄 AI Invoice Assistant 🚀
    </h1>
    """, unsafe_allow_html=True
)

st.sidebar.header("⚙️ Configuration")

# --- API Key Handling ---
# 1. Try to get API key from st.secrets first (for deployment)
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
# Always attempt to configure if an effective API key is present.
# This handles initial setup and cases where the user updates the key.
if effective_gemini_api_key:
    try:
        genai.configure(api_key=effective_gemini_api_key)
        # st.sidebar.success("✅ Gemini API configured.") # No need to show this success, sidebar success is for admin
    except Exception as e:
        st.sidebar.error(f"❌ Failed to configure Gemini API: {e}. Please check your key.")
        effective_gemini_api_key = "" # Clear key if configuration fails to prevent re-attempts with bad key


# Define the password for processing
# Access it from secrets.toml. If not found, fall back to "Rajeev".
ACCESS_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "Rajeev") 

# Changed label here
user_entered_password = st.sidebar.text_input("Enter Password for Admin Panel:", type="password")

# Display admin panel activated message
if user_entered_password == ACCESS_PASSWORD and user_entered_password != "":
    st.sidebar.success("🎉 Admin Panel Activated!")


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

col1, col2 = st.columns([0.6, 0.4])

with col1:
    process_button = st.button("🚀 Process Invoices", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear Results", use_container_width=True)

if clear_button:
    st.session_state.summary_rows = []
    # Force a rerun to clear the file uploader visually and state
    st.experimental_rerun()

if process_button:
    # --- Corrected API Key Check (uses effective_gemini_api_key) ---
    if not effective_gemini_api_key: # Check the effective_gemini_api_key
        st.error("❗ Please enter your Gemini API Key in the sidebar.")
    elif user_entered_password != ACCESS_PASSWORD: # Compare against the defined ACCESS_PASSWORD
        st.error("🔒 Incorrect password. Please enter the correct password to proceed.")
    elif not uploaded_files:
        st.error("⬆️ Please upload at least one PDF file to process.")
    elif not gemini_model_id_input:
        st.error("💡 Please specify a Gemini Model ID in the sidebar.")
    else:
        st.session_state.summary_rows = [] # Clear previous results
        status_text = st.empty() # Placeholder for status messages
        progress_bar = st.progress(0) # Placeholder for overall progress
        total_files = len(uploaded_files)

        for i, uploaded_file_obj in enumerate(uploaded_files):
            st.markdown("---") # Separator for each file's processing
            
            # Use status_text for overall file processing updates
            status_text.info(f"⏳ Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files})")
            
            temp_file_path = None
            
            def update_progress(percentage, message):
                """Callback function to update the progress bar and status message."""
                progress_bar.progress(int(percentage * 100))
                # Append file-specific message to the status text
                status_text.info(f"⏳ Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files})\n\n{message}")


            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file_obj.getvalue())
                    temp_file_path = tmp.name

                # Extract data using the Gemini API
                extracted_data = extract_structured_data(
                    gemini_model_id=gemini_model_id_input,
                    file_path=temp_file_path,
                    pydantic_schema=Invoice,
                    progress_callback=update_progress
                )

                if extracted_data:
                    st.success(f"🎉 Successfully extracted data from **{uploaded_file_obj.name}**")
                    
                    # Handle potential None values for display gracefully
                    cgst = extracted_data.cgst_amount if extracted_data.cgst_amount is not None else 0.0
                    sgst = extracted_data.sgst_amount if extracted_data.sgst_amount is not None else 0.0
                    igst = extracted_data.igst_amount if extracted_data.igst_amount is not None else 0.0
                    # Fallback calculation if total_tax_amount is None
                    total_tax = extracted_data.total_tax_amount if extracted_data.total_tax_amount is not None else (cgst + sgst + igst) 
                    pos = extracted_data.place_of_supply if extracted_data.place_of_supply else "N/A"
                    buyer_gstin_display = extracted_data.buyer_gstin or "N/A"
                    expense_ledger = extracted_data.expense_ledger_suggestion or "Uncategorized"
                    tds_status = extracted_data.tds_applicability or "Not Specified"
                    rcm_status = extracted_data.rcm_applicability or "Not Specified"

                    # Create a more detailed narration including line items
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
                st.exception(e_outer) # Display full traceback for any unhandled exceptions
            finally:
                # Ensure the temporary file is deleted
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
            # Update overall progress bar after each file
            progress_bar.progress((i + 1) / total_files)
            time.sleep(0.1) # Small delay for animation effect

        st.markdown("---")
        if st.session_state.summary_rows:
            st.balloons() # Celebrate successful extraction
            status_text.success("✅ All invoices processed successfully!")
        else:
            status_text.warning("No data extracted from any of the uploaded invoices.")

# Display extracted summary table if available
if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    
    # Format financial columns for better display in the Streamlit dataframe
    financial_cols = ["Total Gross Worth (₹)", "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Tax (₹)", "Total Payable (₹)"]
    for col in financial_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"₹{float(x):,.2f}" if pd.notnull(x) and x != '' else "N/A")

    st.dataframe(df, use_container_width=True)

    # Provide download link for Excel
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # For Excel download, convert financial columns back to numeric for proper Excel calculations
        df_for_excel = pd.DataFrame(st.session_state.summary_rows)
        for col in financial_cols:
            if col in df_for_excel.columns:
                # Remove '₹' and convert to numeric, coerce errors to NaN
                df_for_excel[col] = pd.to_numeric(df_for_excel[col].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce') 

        df_for_excel.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the extracted invoice data as an Excel spreadsheet."
    )
elif not uploaded_files and not process_button: # Only show this guidance if nothing has been uploaded or processed yet
    st.info("Upload PDF files and click '🚀 Process Invoices' to see results.")

st.markdown("---")
st.caption("Developed using Gemini AI and Streamlit.")
