import pdfplumber
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re 
import os

load_dotenv()

def simple_text_splitter(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def clean_pdf_text(text):
    """Clean extracted PDF text to fix formatting issues"""
    
    # Fix with asterisks: 35millionin2019**to**50million
    text = re.sub(r'(\d+)millionin(\d{4})\*\*to\*\*(\d+)', r'$\1 million in \2 to $\3', text)
    
    # Fix with Unicode asterisks: 35millionin2019∗∗to∗∗50million
    text = re.sub(r'(\d+)millionin(\d{4})∗∗to∗∗(\d+)', r'$\1 million in \2 to $\3', text)
    
    # Fix complex pattern: 35millionin2019to50million
    text = re.sub(r'(\d+)millionin(\d{4})to(\d+)million', r'$\1 million in \2 to $\3 million', text)
    
    # Fix: 35millionin2019 → $35 million in 2019
    text = re.sub(r'(\d+)millionin(\d{4})', r'$\1 million in \2', text)
    
    # Fix remaining: 35million → $35 million
    text = re.sub(r'(?<!\$)(\d+)million', r'$\1 million', text)
    
    # Fix: 35billion → $35 billion
    text = re.sub(r'(?<!\$)(\d+)billion', r'$\1 billion', text)
    
    # Fix: to**50 or to∗∗50 → to $50
    text = re.sub(r'to\*\*(\d+)', r'to $\1', text)
    text = re.sub(r'to∗∗(\d+)', r'to $\1', text)
    
    # Clean up remaining asterisks
    text = text.replace('**', ' ')
    text = text.replace('∗∗', ' ')
    
    # Fix double dollar signs
    text = text.replace('$$', '$')
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


# Load PDF
print(" Loading PDF...")
pdf = pdfplumber.open('data/Company_Brochure.pdf')
full_text = ''
for page in pdf.pages:
    text = page.extract_text()
    if text:
        full_text += text
pdf.close()

# Check raw PDF text
print("\n CHECKING RAW PDF TEXT...")
if "35million" in full_text:
    idx = full_text.find("35million")
    print(f" Found issue at position {idx}")
    print(f"Raw snippet: '{full_text[max(0,idx-50):idx+150]}'")
else:
    print(" No '35million' found in raw PDF")

print(" Cleaning text...")
cleaned_text = clean_pdf_text(full_text)

#  Check cleaned text
print("\n CHECKING CLEANED TEXT...")
if "35million" in cleaned_text:
    idx = cleaned_text.find("35million")
    print(f" Still there at position {idx}")
    print(f"Cleaned snippet: '{cleaned_text[max(0,idx-50):idx+150]}'")
else:
    print("Successfully cleaned!")
    # Find and show the fixed version
    if "$35 million" in cleaned_text:
        idx = cleaned_text.find("$35 million")
        print(f"Fixed version found at position {idx}")
        print(f"Fixed snippet: '{cleaned_text[max(0,idx-50):idx+150]}'")

full_text = cleaned_text

# Split into chunks
print(" Splitting text...")
text_chunks = simple_text_splitter(full_text, 1000, 100)

# Create Document objects
documents = []
for i, chunk in enumerate(text_chunks):
    documents.append(Document(
        page_content=chunk,
        metadata={'source': 'data/Company_Brochure.pdf', 'chunk': i}
    ))

# Create embeddings and FAISS index
print(" Creating FAISS index...")
embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model='text-embedding-3-small'
)
vectorstore = FAISS.from_documents(documents, embeddings)

# Save
print(" Saving...")
vectorstore.save_local('data/faiss_index')

print(f" FAISS index recreated with {len(documents)} chunks!")