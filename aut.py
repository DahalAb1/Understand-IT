import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
import tempfile
import streamlit as st

# Loading FLAN-T5 Large ( Model I've used) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Helper Functions

def extract_text_from_pdf(file):
    """Extract text from all pages of a PDF."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n\n"
    return text.strip()

def chunk_text(text, max_tokens=512):
    """Split text into manageable chunks that fit the model input size."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        potential_chunk = current_chunk + sentence + ". "
        if len(tokenizer(potential_chunk)["input_ids"]) < max_tokens:
            current_chunk = potential_chunk
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def reword_text(text, level):
    """Reword the entire input text chunk by chunk."""
    if level == "Basic":
        instruction = "Rewrite this using very simple vocabulary:\n"
    elif level == "Intermediate":
        instruction = "Rewrite this in moderately simple language:\n"
    else:
        instruction = "Rewrite this professionally using rich vocabulary:\n"

    chunks = chunk_text(text)
    rewritten_chunks = []

    for chunk in chunks:
        prompt = instruction + chunk
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda" if torch.cuda.is_available() else "cpu")
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        rewritten = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        rewritten_chunks.append(rewritten)

    return "\n\n".join(rewritten_chunks)

def clean_text_for_pdf(text):
    """Remove characters that can't be encoded in Latin-1."""
    return ''.join(c if 0 <= ord(c) <= 255 else '?' for c in text)

def text_to_pdf(text):
    """Convert text to a PDF file."""
    text = clean_text_for_pdf(text)
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

# Streamlit App UI 

st.set_page_config(page_title="Understand IT", layout="centered")
st.title("Understand IT")
st.write("Upload a PDF and get it reworded in simpler or more professional language.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    input_text = extract_text_from_pdf(uploaded_file)

    if not input_text.strip():
        st.error("No readable text found in the PDF.")
    else:
        st.subheader("Choose Rewording Level:")
        level = st.radio("Select the complexity:", ("Basic", "Intermediate", "Advanced"))

        if st.button("Reword Full Document"):
            with st.spinner("Processing..."):
                reworded_text = reword_text(input_text, level)

            st.subheader("Reworded Output")
            st.write(reworded_text)

            st.subheader("Download Options")
            st.download_button(
                label="Download as .txt",
                data=reworded_text,
                file_name="reworded_text.txt",
                mime="text/plain"
            )

            pdf_file = text_to_pdf(reworded_text)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="Download as .pdf",
                    data=f.read(),
                    file_name="reworded_text.pdf",
                    mime="application/pdf"
                )
else:
    st.info("Please upload a PDF file to begin.")
