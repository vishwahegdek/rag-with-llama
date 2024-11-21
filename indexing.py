import os
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
import pdf_extract

# Load API keys securely
PINECONE_API_KEY = "pcsk_4omisS_3rPyuD2Xnu8Bwx7hTCWrQ4uAnnSD9GNmjb8NttbJVXF2VzRJP8M6TpYRiAorvNW"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name and check if it exists; create if not
INDEX_NAME = "pdf-knowledge-base"
if INDEX_NAME not in pc.list_indexes():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # This is typical dimension for BERT
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Pinecone index object
index = pc.Index(name=INDEX_NAME)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    """
    Get embedding for a given text using BERT model and convert it to list.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    return mean_pooled_embeddings[0].numpy().tolist()  # Convert ndarray to list




def save_to_pinecone(chunks, metadata, batch_size=50):
    """
    Save text chunks into Pinecone with associated metadata, ensuring proper serialization.
    This function now processes the upserts in batches to avoid exceeding API limits.
    """
    upsert_data = []
    total_chunks = len(chunks)
    batches = (total_chunks + batch_size - 1) // batch_size  # Calculate the number of batches

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)  # This now returns a list
        upsert_data.append((f"{metadata['doc_id']}-{i}", embedding, {**metadata, "chunk_id": i, "text": chunk}))

        # Check if we've reached the batch size or the end of the list
        if (i + 1) % batch_size == 0 or i + 1 == total_chunks:
            index.upsert(upsert_data)
            print(f"Saved batch of {len(upsert_data)} chunks to Pinecone.")
            upsert_data = []  # Reset for next batch



# Example usage: index some chunks with metadata
metadata = {
    "doc_id": "1",
    "author": "Vishwa",
    "topic": "NLP"
}
chunks = pdf_extract.chunks
save_to_pinecone(chunks, metadata)

print("Data indexed successfully!")
