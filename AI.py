import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import os
import tempfile # For temporary file handling
import io # For Excel download
import requests # For loading Lottie animations
from streamlit_lottie import st_lottie # For displaying Lottie animations
import traceback # For detailed error reporting

# Try to import google.generativeai, show error if not found
try:
    from google import genai
except ImportError:
    st.error("The 'google-generativeai' library is not installed. Please install it by running: pip install google-generativeai")
    st.stop() # Keep this stop as it's a fundamental library import check

st.set_page_config(layout="wide")

# --- Lottie animations for better UX ---
# URLs for Lottie animation JSON files
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    """Loads Lottie animation JSON safely from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation from {url}: {e}")
        return None

# Load Lottie JSON data
hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# --- Initial UI setup and Lottie "hello" animation ---
# Initialize session state for tracking file uploads and processing
if "files_uploaded" not in st.session_state:
    st.session_state["files_uploaded"] = False
if "process_triggered" not in st.session_state:
    st.session_state["process_triggered"] = False
if "file_uploader_key" not in st.session_state: # Key for resetting file uploader
    st.session_state["file_uploader_key"] = 0
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
if 'client' not in st.session_state: # Client will be initialized on process button click
    st.session_state.client = None

# Display initial Lottie animation if no files have been uploaded yet
# and no processing has been triggered (i.e., fresh start or after clear)
if not st.session_state["files_uploaded"] and not st.session_state["process_triggered"] and not st.session_state.summary_rows:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 PDF Invoice Extractor (Gemini AI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using Google Gemini AI.")
st.markdown("---")

# --- Admin Passcode and Gemini API Key Configuration (inputs moved here) ---
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev" # Define your admin password here

# Determine the API key source but don't stop the app yet if it's missing
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted. Using API key from Streamlit secrets.")
    gemini_api_key_candidate = st.secrets.get("GEMINI_API_KEY") # Candidate API key
    if not gemini_api_key_candidate:
        st.sidebar.error("`GEMINI_API_KEY` missing in Streamlit secrets. Please configure it for admin access.")
else:
    gemini_api_key_candidate = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
    if not gemini_api_key_candidate:
        st.sidebar.warning("Please enter a valid API key in the sidebar to process invoices.")

# Default Gemini model ID input (moved up to be always visible)
DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
gemini_model_id_input = st.sidebar.text_input("Gemini Model ID for Extraction:", DEFAULT_GEMINI_MODEL_ID)
st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure the model ID is correct and supports schema-based JSON output.")


# --- Pydantic Models (as provided - no changes) ---
class LineItem(BaseModel):
    description: str
    quantity: float
    gross_worth: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    gstin: str
    seller_name: str
    buyer_name: str
    buyer_gstin: Optional[str] = None
    line_items: List[LineItem]
    total_gross_worth: float
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    place_of_supply: Optional[str] = None
    expense_ledger: Optional[str] = None
    tds: Optional[str] = None

# --- Helper function to get the Gemini-compatible schema ---
def get_invoice_schema_for_gemini():
    """Returns a dictionary representing the Invoice Pydantic model's schema in a Gemini-compatible format."""
    return {
        "type": "OBJECT",
        "properties": {
            "invoice_number": {"type": "STRING"},
            "date": {"type": "STRING"},
            "gstin": {"type": "STRING"},
            "seller_name": {"type": "STRING"},
            "buyer_name": {"type": "STRING"},
            "buyer_gstin": {"type": "STRING"},
            "line_items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "description": {"type": "STRING"},
                        "quantity": {"type": "NUMBER"},
                        "gross_worth": {"type": "NUMBER"}
                    }
                }
            },
            "total_gross_worth": {"type": "NUMBER"},
            "cgst": {"type": "NUMBER"},
            "sgst": {"type": "NUMBER"},
            "igst": {"type": "NUMBER"},
            "place_of_supply": {"type": "STRING"},
            "expense_ledger": {"type": "STRING"},
            "tds": {"type": "STRING"}
        }
    }


# --- Gemini API Interaction Function (FIXED) ---
def extract_structured_data(
    client_instance, # The genai.Client object (unused for model interaction now but passed for consistency)
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel, # Still passed, but schema used is from get_invoice_schema_for_gemini()
):
    display_name = os.path.basename(file_path)
    gemini_file_resource = None # To store the Gemini File API object for deletion

    try:
        # 1. Upload the file to the File API
        st.write(f"Uploading '{display_name}' to Gemini File API...")
        gemini_file_resource = client_instance.files.upload(
            file=file_path,
            config={'display_name': display_name.split('.')[0]}
        )
        st.write(f"'{display_name}' uploaded. Gemini file name: {gemini_file_resource.name}")

        # 2. Generate a structured response using the Gemini API
        prompt = (
            "Extract all relevant and clear information from the invoice, adhering to Indian standards "
            "for dates (DD/MM/YYYY or DD-MM-YYYY) and codes (like GSTIN, HSN/SAC). "
            "Accurately identify the total amount payable. Classify the nature of expense and suggest an "
            "applicable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). "
            "Determine TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain'). "
            "Determine reverse charge GST (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain'). "
            "Handle missing data appropriately by setting fields to null or an empty string where "
            "Optional, and raise an issue if critical data is missing for required fields. "
            "Do not make assumptions or perform calculations beyond what's explicitly stated in the invoice text. "
            "If a value is clearly zero, represent it as 0.0 for floats. For dates, prefer DD/MM/YYYY."
        )
        st.write(f"Sending '{display_name}' to Gemini model '{gemini_model_id}' for extraction...")

        # --- FIX APPLIED HERE: Instantiate GenerativeModel directly and use custom schema ---
        # The genai.GenerativeModel takes the model_id and manages the API call.
        # It needs the API key to be set either via genai.configure() or environment variable.
        # Since genai.configure() is called when the Process button is clicked, this should work.
        model = genai.GenerativeModel(gemini_model_id)

        response = model.generate_content(
            contents=[prompt, gemini_file_resource],
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': get_invoice_schema_for_gemini() # Use the custom, Gemini-compatible schema
            }
        )
        # --- END FIX ---

        st.write(f"Data extracted for '{display_name}'.")
        return response.parsed

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
    finally:
        # 3. Clean up: Delete the file from Gemini File API if it was uploaded
        if gemini_file_resource and hasattr(client_instance, 'files') and hasattr(client_instance.files, 'delete'):
            try:
                st.write(f"Attempting to delete '{gemini_file_resource.name}' from Gemini File API...")
                client_instance.files.delete(name=gemini_file_resource.name) # Assumes this method exists
                st.write(f"Successfully deleted '{gemini_file_resource.name}' from Gemini.")
            except Exception as e_del:
                st.warning(f"Could not delete '{gemini_file_resource.name}' from Gemini File API: {e_del}")
        elif gemini_file_resource:
            st.warning(f"Could not determine how to delete Gemini file '{gemini_file_resource.name}'. "
                        "Manual cleanup may be required in your Gemini project console.")


# --- Instructions and File Uploader ---
st.info(
    "**Instructions:**\n"
    "1. Enter the Admin Passcode or your Gemini API Key in the sidebar.\n"
    "2. Optionally, change the Gemini Model ID if needed.\n"
    "3. Upload one or more PDF invoice files.\n"
    "4. Click 'Process Invoices' to extract data.\n"
    "   The extracted data will be displayed in a table and available for download as Excel."
)

# Placeholders for dynamic content, including the file uploader
file_uploader_placeholder = st.empty()

with file_uploader_placeholder.container():
    uploaded_files = st.file_uploader(
        "📤 Upload scanned invoice PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    # Store uploaded files in session state to persist across reruns and allow clearing
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["files_uploaded"] = True
    else:
        st.session_state["uploaded_files"] = []
        st.session_state["files_uploaded"] = False


# Conditional display of buttons after file upload or if results exist
if st.session_state["files_uploaded"] or st.session_state.summary_rows:
    col_process, col_spacer, col_clear = st.columns([1, 4, 1])

    with col_process:
        if st.button("🚀 Process Invoices", type="primary", help="Click to start extracting data from uploaded invoices."):
            st.session_state["process_triggered"] = True
            st.session_state.summary_rows = [] # Clear previous results on new processing run

            # --- API Key Validation and Client Initialization (moved inside button click) ---
            if not gemini_api_key_candidate:
                st.error("Please provide a Gemini API Key in the sidebar to process invoices.")
                st.session_state["process_triggered"] = False # Do not proceed with processing
                st.stop() # Stop further execution for this run
            
            try:
                # Configure the genai module with the API key
                genai.configure(api_key=gemini_api_key_candidate)
                # Initialize Gemini client for file operations if needed (e.g., upload/delete)
                st.session_state.client = genai.Client(api_key=gemini_api_key_candidate)
                st.info("Processing initiated. Please wait...")
            except Exception as e:
                st.error(f"Failed to initialize Gemini client with the provided key: {e}")
                st.session_state.client = None # Reset client on failure
                st.session_state["process_triggered"] = False # Do not proceed
                st.stop() # Stop further execution for this run


    with col_clear:
        if st.button("🗑️ Clear All Files & Reset", help="Click to clear all uploaded files and extracted data."):
            # Increment key BEFORE clearing session_state to ensure uploader reset
            st.session_state["file_uploader_key"] += 1

            # Clear all relevant session state variables explicitly
            st.session_state["files_uploaded"] = False
            st.session_state.summary_rows = [] # Clear summary results
            st.session_state["process_triggered"] = False
            st.session_state["uploaded_files"] = [] # Explicitly empty this list
            st.session_state.client = None # Reset client state

            # Clear the placeholder to remove the old file uploader instance
            file_uploader_placeholder.empty()

            # This rerun will redraw everything, including a *new* file uploader with the incremented key
            st.rerun()

# Only proceed with processing if files are uploaded AND the "Process Invoices" button was clicked AND client is initialized
if st.session_state["uploaded_files"] and st.session_state["process_triggered"] and st.session_state.client:
    total_files = len(st.session_state["uploaded_files"])
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, uploaded_file_obj in enumerate(st.session_state["uploaded_files"]):
        file_name = uploaded_file_obj.name
        progress_text.text(f"Processing file: {file_name} ({i+1}/{total_files})")
        progress_bar.progress((i + 1) / total_files)

        st.markdown(f"**Current File: {file_name}**")
        temp_file_path = None
        try:
            # Save UploadedFile to a temporary file to get a file_path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp:
                tmp.write(uploaded_file_obj.getvalue())
                temp_file_path = tmp.name

            with st.spinner(f"🧠 Extracting data from {uploaded_file_obj.name} using Gemini AI..."):
                extracted_data = extract_structured_data(
                    client_instance=st.session_state.client,
                    gemini_model_id=gemini_model_id_input,
                    file_path=temp_file_path,
                    pydantic_schema=Invoice # The Pydantic model for structuring output
                )

            if extracted_data:
                st.success(f"Successfully extracted data from {uploaded_file_obj.name}")
                # Ensure all fields are handled, especially Optionals
                cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                pos = extracted_data.place_of_supply if extracted_data.place_of_supply else "N/A"
                buyer_gstin_display = extracted_data.buyer_gstin or "N/A"
                expense_ledger_display = extracted_data.expense_ledger or "N/A"
                tds_display = extracted_data.tds or "N/A"

                narration = (
                    f"Invoice {extracted_data.invoice_number} dated {extracted_data.date} "
                    f"was issued by {extracted_data.seller_name} (GSTIN: {extracted_data.gstin}) "
                    f"to {extracted_data.buyer_name} (GSTIN: {buyer_gstin_display}), "
                    f"with a total value of ₹{extracted_data.total_gross_worth:.2f}. "
                    f"Taxes applied - CGST: ₹{cgst:.2f}, SGST: ₹{sgst:.2f}, IGST: ₹{igst:.2f}. "
                    f"Place of supply: {pos}. Expense: {expense_ledger_display}. "
                    f"TDS: {tds_display}."
                )
                st.session_state.summary_rows.append({
                    "File Name": uploaded_file_obj.name,
                    "Invoice Number": extracted_data.invoice_number,
                    "Date": extracted_data.date,
                    "Seller Name": extracted_data.seller_name,
                    "Seller GSTIN": extracted_data.gstin,
                    "Buyer Name": extracted_data.buyer_name,
                    "Buyer GSTIN": buyer_gstin_display,
                    "Total Gross Worth": extracted_data.total_gross_worth,
                    "CGST": cgst,
                    "SGST": sgst,
                    "IGST": igst,
                    "Place of Supply": pos,
                    "Expense Ledger": expense_ledger_display,
                    "TDS": tds_display,
                    "Narration": narration,
                })
            else:
                st.warning(f"Failed to extract data or no data returned for {uploaded_file_obj.name}")

        except Exception as e_outer:
            st.error(f"An unexpected error occurred while processing {uploaded_file_obj.name}: {e_outer}")
            st.text_area(f"Error Details for {uploaded_file_obj.name}", traceback.format_exc(), height=150)
        finally:
            # Clean up: Delete the temporary local file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                st.write(f"Deleted temporary local file: {temp_file_path}")
    progress_bar.empty()
    progress_text.empty()
    st.markdown(f"---")
    # Display completion animation and balloons if successful
    if st.session_state.summary_rows:
        if completed_json:
            st_lottie(completed_json, height=200, key="done_animation")
        st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices Processed!!! 😊</h3>", unsafe_allow_html=True)
        st.balloons()


if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)

    # Reorder columns for display
    display_cols = [
        "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN",
        "Buyer Name", "Buyer GSTIN", "Total Gross Worth", "CGST", "SGST", "IGST",
        "Place of Supply", "Expense Ledger", "TDS", "Narration"
    ]
    # Ensure all display columns exist in the DataFrame, add missing ones with default values
    for col in display_cols:
        if col not in df.columns:
            df[col] = "" # Or appropriate default like 0.0 for numbers

    st.dataframe(df[display_cols], hide_index=True, use_container_width=True)

    # Provide download link for Excel
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not st.session_state["files_uploaded"] and not st.session_state["process_triggered"]:
    st.info("Upload PDF files and click 'Process Invoices' to see results.")
elif st.session_state["files_uploaded"] and not st.session_state["process_triggered"]:
    st.info("Files uploaded. Click 'Process Invoices' to start extraction.")
