import torch 
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
import tempfile
from cache import get as cache_get, set_cache as cache_set


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")




#extract text from pdf 
def extract_text_from_pdf(file):
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


#reword text 
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

        # checking the cache before executing the model 
        cached = cache_get(chunk)
        if cached:
            rewritten_chunks.append(cached)
            continue 
        
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

        #storing any new result in cache so same solution is produced instantly 

        cache_set(chunk,rewritten)

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