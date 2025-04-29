# Understand‑IT
#### A technology that rephrases language to make ideas clearer and more accessible to a broader audience.
---

# Understand-It

## What Is Understand-It?

Legal agreements, academic papers, and even everyday instructions are often written in overly complex, jargon-heavy language. But this complexity is rarely necessary. Most information can be conveyed just as effectively using a simpler, more understandable set of words—without losing meaning or precision.

Unnecessary complexity:

- Wastes valuable human hours  
- Makes education less accessible  
- Raises barriers to entry for many people  

**Understand-It** is a technology designed to decode dense, complicated content—such as contracts, disclosures, academic papers, legal documents, or any kind of text—into clear, accessible language. By simplifying communication, it helps address these challenges and contributes to a more informed, inclusive, and empowered society.

---

### Example of Simplification

**Original (Complex):**  
*"The lessee shall indemnify and hold harmless the lessor from any and all liabilities, claims, and demands, whether arising in tort or contract, which may result from the lessee’s occupancy or use of the leased premises."*

**Simplified Version:** 
*"The person renting must protect the owner from any problems, claims, or lawsuits that happen because of their use of the property."*

---

By making language easier to understand, Understand-It empowers more people to confidently engage with information that might otherwise seem intimidating or confusing.

 
 ---



---

## 2.  Future improvements

- **Migrate to mobile device:** I plan to migrate the application to mobile platforms using Flutter, this idea is better suited for mobile phones. 

- **Include my story behind how I came to this idea**: Add a section in this repo "The spark behind Understand-It" 
  
- **Text-to-Speech (TTS):** This will make text accessible to people with hearing disabilities or anyone who would require speech. 

- **Image-to-Text (OCR):**  This feature will allow users to capture text from images and have it paraphrased.

- **Speech-to-Text Input:**  Allows users to input text via speech, which will then be transcribed and paraphrased.

- **Readability Meter:**  This will provide users with insights into the complexity of the text and the effectiveness of the paraphrasing.

- **Model Selection:**  Use better model than gpt-neo. Also, change how the model loads information because right now it's very heavy on storage. Basically the main purpose of this is to improve user access and user experience, make it faster and more reliable.

---

 ## 3. Overview of Current Model 

| Module                 | Purpose                                                        | Highlights                                           |
|------------------------|----------------------------------------------------------------|------------------------------------------------------|
| `extract_text_from_pdf` | Pulls text from the first page of an uploaded PDF             | Built on `pypdf` (lightweight, no external server)   |
| `reword_text`          | Sends a prompt to GPT-Neo 1.3B to rewrite text at three levels: Basic, Intermediate, Advanced | Open-source weights available via Hugging Face       |
| `text_to_pdf`          | Outputs a cleaned, one-page PDF of the rewritten text         | Handles common Latin-1 encoding errors in FPDF       |
| **Streamlit UI**       | Basic front end UI, live output & download buttons | Uses the standard `st.file_uploader` widget          |


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
| `torch` | Deep learning framework | Not any specific puspose here, Installing would help in debugging |
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

#### 7. Rest of the code is streamlist front end 


