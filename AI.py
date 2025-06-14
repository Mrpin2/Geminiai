import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import os
import tempfile # For temporary file handling
import io # For Excel download

# Try to import google.generativeai, show error if not found
try:
    from google import genai
except ImportError:
    st.error("The 'google-generativeai' library is not installed. Please install it by running: pip install google-generativeai")
    st.stop()

# --- Pydantic Models (as provided) ---
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

# --- Gemini API Interaction Function ---
# This function is adapted from your script
# It assumes 'client_instance' has 'files.upload' and 'models.generate_content' methods
def extract_structured_data(
    client_instance, # The genai.Client object
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
):
    display_name = os.path.basename(file_path)
    gemini_file_resource = None # To store the Gemini File API object for deletion

    try:
        # 1. Upload the file to the File API
        st.write(f"Uploading '{display_name}' to Gemini File API...")
        # This uses the client.files.upload structure from your script
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
        # This uses the client.models.generate_content structure from your script
        response = client_instance.models.generate_content(
            model=gemini_model_id,
            contents=[prompt, gemini_file_resource],
            config={'response_mime_type': 'application/json', 'response_schema': pydantic_schema}
        )

        st.write(f"Data extracted for '{display_name}'.")
        # Convert the response to the pydantic model and return it
        # Your original script uses response.parsed
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


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📄 PDF Invoice Extractor (Gemini AI)")

st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# Use a known good model, but allow override if user is sure about "gemini-2.0-flash"
# The model "gemini-2.0-flash" from your script might be an internal or non-public model.
# "gemini-1.5-flash-latest" is a good public alternative for this task.
DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
gemini_model_id_input = st.sidebar.text_input("Gemini Model ID for Extraction:", DEFAULT_GEMINI_MODEL_ID)
st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Your script used 'gemini-2.0-flash'. "
                   "Ensure the model ID is correct and supports schema-based JSON output.")


st.info(
    "**Instructions:**\n"
    "1. Enter your Gemini API Key in the sidebar.\n"
    "2. Optionally, change the Gemini Model ID if needed.\n"
    "3. Upload one or more PDF invoice files.\n"
    "4. Click 'Process Invoices' to extract data.\n"
    "   The extracted data will be displayed in a table and available for download as Excel."
)

uploaded_files = st.file_uploader(
    "Choose PDF invoice files",
    type="pdf",
    accept_multiple_files=True
)

if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
if 'client' not in st.session_state:
    st.session_state.client = None


if st.button("🚀 Process Invoices", type="primary"):
    if not api_key_input:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not gemini_model_id_input:
        st.error("Please specify a Gemini Model ID in the sidebar.")
    else:
        try:
            # Initialize client here, as API key might change or processing is requested
            st.session_state.client = genai.Client(api_key=api_key_input)
            st.success("Gemini client initialized successfully with the provided API Key.")
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {e}")
            st.session_state.client = None # Reset client on failure

        if st.session_state.client:
            st.session_state.summary_rows = [] # Clear previous results
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for i, uploaded_file_obj in enumerate(uploaded_files):
                st.markdown(f"---")
                st.info(f"Processing file: {uploaded_file_obj.name} ({i+1}/{total_files})")
                temp_file_path = None
                try:
                    # Save UploadedFile to a temporary file to get a file_path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp:
                        tmp.write(uploaded_file_obj.getvalue())
                        temp_file_path = tmp.name

                    with st.spinner(f"Extracting data from {uploaded_file_obj.name}..."):
                        extracted_data = extract_structured_data(
                            client_instance=st.session_state.client,
                            gemini_model_id=gemini_model_id_input,
                            file_path=temp_file_path,
                            pydantic_schema=Invoice # The Pydantic model for structuring output
                        )

                    if extracted_data:
                        st.success(f"Successfully extracted data from {uploaded_file_obj.name}")
                        cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                        sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                        igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                        pos = extracted_data.place_of_supply if extracted_data.place_of_supply else "N/A"
                        buyer_gstin_display = extracted_data.buyer_gstin or "N/A"

                        narration = (
                            f"Invoice {extracted_data.invoice_number} dated {extracted_data.date} "
                            f"was issued by {extracted_data.seller_name} (GSTIN: {extracted_data.gstin}) "
                            f"to {extracted_data.buyer_name} (GSTIN: {buyer_gstin_display}), "
                            f"with a total value of ₹{extracted_data.total_gross_worth:.2f}. "
                            f"Taxes applied - CGST: ₹{cgst:.2f}, SGST: ₹{sgst:.2f}, IGST: ₹{igst:.2f}. "
                            f"Place of supply: {pos}. Expense: {extracted_data.expense_ledger or 'N/A'}. "
                            f"TDS: {extracted_data.tds or 'N/A'}."
                        )
                        st.session_state.summary_rows.append({
                            "File Name": uploaded_file_obj.name,
                            # "File Path": temp_file_path, # Temp path might not be useful long term
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
                            "Expense Ledger": extracted_data.expense_ledger,
                            "TDS": extracted_data.tds,
                            "Narration": narration,
                        })
                    else:
                        st.warning(f"Failed to extract data or no data returned for {uploaded_file_obj.name}")

                except Exception as e_outer:
                    st.error(f"An unexpected error occurred while processing {uploaded_file_obj.name}: {e_outer}")
                finally:
                    # Clean up: Delete the temporary local file
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        st.write(f"Deleted temporary local file: {temp_file_path}")
                progress_bar.progress((i + 1) / total_files)

            st.markdown(f"---")
            if st.session_state.summary_rows:
                st.balloons()


if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    st.dataframe(df)

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
elif not uploaded_files:
     st.info("Upload PDF files and click 'Process Invoices' to see results.")
