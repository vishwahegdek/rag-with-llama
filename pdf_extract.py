from PyPDF2 import PdfReader
import re
import nltk
from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(pdf_path, start_page=0, end_page=3):
    """
    Extracts text from a range of pages in a PDF.
    Args:
        pdf_path (str): Path to the PDF file.
        start_page (int): Starting page index (0-based).
        end_page (int): Ending page index (exclusive, 0-based).
    Returns:
        str: Extracted text from the specified range.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for i in range(start_page, min(end_page, len(reader.pages))):
        text += reader.pages[i].extract_text() or ""
    return text


# Download NLTK resources
nltk.download('punkt_tab')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(raw_text):
    """
    Clean and preprocess raw text.
    - Remove special characters, extra spaces, and irrelevant content.
    - Split text into coherent chunks.
    """
    # Remove special characters
    cleaned_text = re.sub(r'[^\w\s.,]', '', raw_text)
    
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def chunk_text(cleaned_text, chunk_size=1000):
    """
    Split the cleaned text into smaller chunks for embedding.
    """
    sentences = sent_tokenize(cleaned_text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(" ".join(current_chunk)) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    # Append the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Example usage
raw_text = extract_text_from_pdf("MSME_Schemes_English_0.pdf",0,265)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text, chunk_size=1200)

# Print the chunks
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")