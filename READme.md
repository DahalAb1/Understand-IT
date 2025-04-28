# Understand‑IT

A technology that paraphrases words or text so everyone—including autistic users, non‑native speakers, and busy professionals—can get the insight the words are trying to express easily.

---

## 1.  Why understand-it

Legal agreements, academic papers, and even everyday instructions are often written in language in overly complex and jargonish structure. Everything can be understood with simpler and understandable words, so I don't see the point in underlying complixity. This just waste productive human hours and make education less accessible, and increases barrier of entry for many people.  Also, this AI‑powered text simplifier, can help customers decode dense contracts information, disclosures, or marketing copy—improving transparency and trust. 

 Using words and language a person is comfortable with:
 - reduces stress
 -  broadens access of information
 - Increase inclusion for people with leanring differences any problem with reading or understanding information.  
 ---

 ## 2. What the Prototype Already Does

| Module                 | Purpose                                                        | Highlights                                           |
|------------------------|----------------------------------------------------------------|------------------------------------------------------|
| `extract_text_from_pdf` | Pulls text from the first page of an uploaded PDF             | Built on `pypdf` (lightweight, no external server)   |
| `reword_text`          | Sends a prompt to GPT-Neo 1.3B to rewrite text at three levels: Basic, Intermediate, Advanced | Open-source weights available via Hugging Face       |
| `text_to_pdf`          | Outputs a cleaned, one-page PDF of the rewritten text         | Handles common Latin-1 encoding errors in FPDF       |
| **Streamlit UI**       | Drag-and-drop PDF uploader, radio-button complexity selector, live output & download buttons | Uses the standard `st.file_uploader` widget          |

---

## 3. Roadmap & Technical Direction

#### 3.1 Near-Term Enhancements

- **Text-to-Speech (TTS):** This will make text accessible to people with hearing disabilities or anyone who would require speech. 

- **Image-to-Text (OCR):**  This feature will allow users to capture text from images and have it paraphrased.

- **Speech-to-Text Input:**  Allows users to input text via speech, which will then be transcribed and paraphrased.

- **Readability Meter:**  This will provide users with insights into the complexity of the text and the effectiveness of the paraphrasing.

- **Model Selection:**  Use better model than gpt-neo. Also, change how the model loads information because right now it's very heavy on storage. Basically the main purpose of this is to improve user access and user experience, make it faster and more reliable.

---
## 4. How to run this program 


#### step 1: Clone the repository 

```bash 
git clone https://github.com/DahalAb1/Understand-IT.git
cd Understand-IT
```

#### step 2: Install Dependencies
```bash 
pip install torch transformers streamlit pypdf fpdf
```
These depndencies are used in the program : torch, transformers, strealit, pypdf,fpdf  


**_you can use pip to install everything_** 


#### Step 3: Run the streamlit application
```bash 
streamlit run app.py
```


### 5. Code Explanation 

### 1. Libraries

```python
import torch
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from fpdf import FPDF
import tempfile
import streamlit as st
```

##### Libraries Used

| Library | Purpose | Why It's Needed |
|---------|---------|-----------------|
| `torch` | Deep learning framework | Not any specific puspose in this code, but may require while working with tensors. |
| `pypdf` | PDF reading and text extraction | Used to extract text from uploaded PDF files (first page). |
| `transformers` | Pre-trained models and tokenizers | Loads GPT-Neo 1.3B model and tokenizer for text paraphrasing. |
| `fpdf` | PDF creation | Generates downloadable PDF files containing the rewritten text. |
| `tempfile` | Temporary file management | Creates temporary files for safe and easy downloads without manual cleanup. |
| `streamlit` | Frontend web framework | Builds the drag-and-drop web application interface for easy user interaction. |


### 2. Extracting Text from PDF 

```python
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    if len(reader.pages) == 0:
        return ""
    text = reader.pages[0].extract_text() or ""
    return text.strip()
```
#### 3. Loading the GPT-Neo 1.3B Model
```python
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```
 - loading pretrained model from hugging face.

### 4. GPT neo pipeline

```python
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
```
- Tokens are inputed into this function so that the model can take in the data and produce resonable output as per the hyperparameters set. 

#### 5. Cleaning Text for PDF 
```python

def clean(text):
    return ''.join(c if 0 <= ord(c) <= 255 else '?' for c in text)

```

- this handles outlier characters and replaces them with?

#### 6. Creating PDF output 

```python

def text_to_pdf(text):
    pdf = FPDF(format="Letter", unit="pt")
    pdf.set_auto_page_break(auto=True, margin=40)
    pdf.set_margins(40, 40, 40)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in clean(text).splitlines():
        pdf.multi_cell(0, 14, line)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name
```
- takes the cleaned text and creates a pdf

#### 7. Rest is streamlist front end

