import streamlit as st
from utils import extract_text_from_pdf
from utils import reword_text
from utils import text_to_pdf



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