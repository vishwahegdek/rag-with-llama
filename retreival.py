import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec


# Load API keys securely
PINECONE_API_KEY = "pcsk_4omisS_3rPyuD2Xnu8Bwx7hTCWrQ4uAnnSD9GNmjb8NttbJVXF2VzRJP8M6TpYRiAorvNW"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


INDEX_NAME = "pdf-knowledge-base"

# Pinecone index object
index = pc.Index(name=INDEX_NAME)

# Assuming the tokenizer and model are loaded and available globally
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


def retrieve_from_pinecone(query, top_k=5):
    """
    Retrieve the top_k most relevant chunks for the query from Pinecone.
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Perform similarity search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    # Return results
    return results

# Assuming 'index' is a global Pinecone index object connected properly
# query = "Related Scheme School dropouts, ITI graduates "  # User's query
# query = "It is for the poorest of the poor (living in the BPL)"  # User's query
# query = "Student"  # User's query

# results = retrieve_from_pinecone(query, top_k=5)

# print("Retrieved Results:")
# for match in results['matches']:
#     print(f"Score: {match['score']}, Text: {match['metadata']['text']}")
