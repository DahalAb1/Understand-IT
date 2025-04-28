import torch
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from fpdf import FPDF
import tempfile
import streamlit as st

# --- Helper Functions ---

def extract_text_from_pdf(file):
    """Extract text from only the first page of a PDF file."""
    reader = PdfReader(file)
    if len(reader.pages) == 0:
        return ""
    
    page = reader.pages[0]
    text = page.extract_text() or ""
    return text.strip()

# --- Load GPT-Neo-1.3B ---
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def reword_text(text, level):
    """Reword the given text based on the selected level using GPT-Neo 1.3B."""
    if level == "Basic":
        prompt = f"Rewrite the following text using very simple vocabulary:\n\n{text}\n\nRewritten Text:"
    elif level == "Intermediate":
        prompt = f"Rewrite the following text using moderately simple vocabulary:\n\n{text}\n\nRewritten Text:"
    else:  # Advanced
        prompt = f"Rewrite the following text professionally using rich vocabulary:\n\n{text}\n\nRewritten Text:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50
    )

    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rewritten = rewritten.replace(prompt, "").strip()
    return rewritten

def clean_text_for_pdf(text):
    """Remove characters that can't be encoded in Latin-1."""
    return ''.join(c if 0 <= ord(c) <= 255 else '?' for c in text)

def text_to_pdf(text):
    """Convert text to a PDF file."""
    text = clean_text_for_pdf(text)  # Clean text before PDF writing

    pdf = FPDF(format="Letter", unit="pt")
    pdf.set_auto_page_break(auto=True, margin=40)
    pdf.set_margins(40, 40, 40)

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    for line in text.splitlines():
        pdf.multi_cell(0, 14, line)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# --- Streamlit App ---

st.set_page_config(page_title="Understand IT", layout="centered")
st.title("Understand IT")

st.write("""
Upload a PDF For paraphrasing.  
""")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    input_text = extract_text_from_pdf(uploaded_file)

    if not input_text.strip():
        st.error("No readable text found.")
    else:
        st.subheader("Choose Your Paraphrasing Level:")
        level = st.radio(
            "Select the rewording complexity:",
            ("Basic", "Intermediate", "Advanced")
        )

        if st.button("Reword Text"):
            with st.spinner("Rewording your text... Please wait."):
                reworded_text = reword_text(input_text, level)

            st.subheader("Reworded Output")
            st.write(reworded_text)

            # --- Download Options ---
            st.subheader("Download Reworded Text")

            st.download_button(
                label="Download as .txt",
                data=reworded_text,
                file_name="reworded_text.txt",
                mime="text/plain"
            )

            pdf_file = text_to_pdf(reworded_text)
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="Download as .pdf",
                data=pdf_bytes,
                file_name="reworded_text.pdf",
                mime="application/pdf"
            )

else:
    st.info("Please upload a PDF file to begin.")
