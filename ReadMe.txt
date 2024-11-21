CashFlow Chatbot - README
Overview
CashFlow Chatbot is a dynamic web application built using Streamlit that interacts with users to provide advice on loan schemes based on their specific criteria. The system employs Retrieval-Augmented Generation (RAG), LLM (LLaMA3.1:8b), and Pinecone for intelligent context retrieval and response generation. It also includes the ability to preprocess and analyze PDF documents for knowledge base creation, embedding generation, and indexing.

Features
Interactive Questionnaire:

Users answer a series of questions to provide input for loan scheme recommendations.
Includes validations for responses and session persistence.
Real-time Chatbot:

Users can chat with an assistant after completing the questionnaire.
The assistant uses RAG to retrieve relevant context and LLM to generate concise, meaningful responses.
PDF Knowledge Base Extraction:

Extracts and preprocesses text from PDF files.
Splits text into chunks for efficient indexing and retrieval.
Knowledge Base Indexing:

Uses Pinecone for vector-based indexing and similarity search.
Embeddings generated with BERT are stored in Pinecone for quick access.
Typing Animation:

Provides a realistic, engaging chatbot interaction.
Tech Stack
Backend Components:
Streamlit: For the interactive user interface.
LangChain & LLaMA: For large language model inference and RAG pipeline.
PyPDF2: For PDF text extraction.
NLTK: For text preprocessing and sentence tokenization.
Transformers (HuggingFace): For embedding generation using the BERT model.
Pinecone: For vector database management and efficient retrieval.
Frontend Features:
Responsive and intuitive interface.
Dynamic input handling for questionnaire responses.
Chatbot interaction with real-time user feedback.
Setup Instructions
Prerequisites
Python 3.8 or later.
Basic understanding of NLP and Streamlit.
Access to a Pinecone API key and LLaMA model.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/<username>/cashflow-chatbot.git
cd cashflow-chatbot
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables:

Create a .env file in the root directory with:
plaintext
Copy code
PINECONE_API_KEY=<your-pinecone-api-key>
Start the LLaMA3.1:8b model server:

bash
Copy code
llama start --base_url http://127.0.0.1:11434
Run the application:

bash
Copy code
streamlit run app.py
Usage
Step 1: Complete the Questionnaire
Answer all questions related to your financial status and occupation.
Save your responses, which are stored in a responses.json file.
Step 2: Chat with the Assistant
Interact with the assistant to receive personalized loan scheme recommendations.
The assistant fetches relevant context using the RAG pipeline.
Step 3: PDF Knowledge Base
Extract, clean, and chunk text from PDFs using the pdf_extract module.
Save the chunks and embeddings to Pinecone for efficient retrieval.
Project Directory Structure
bash
Copy code
cashflow-chatbot/
│
├── app.py                  # Main Streamlit application
├── query_bot.py            # Chatbot logic and response generation
├── retreival.py            # Pinecone retrieval and RAG logic
├── pdf_extract.py          # PDF extraction and chunking utilities
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not included in repo)
└── README.md               # Project documentation
Key Functions
prepare_preprompt:

Combines retrieved context and instructions into a pre-prompt for LLaMA.
extract_text_from_pdf:

Extracts text from a range of pages in a PDF.
clean_text:

Cleans and preprocesses raw text for chunking.
chunk_text:

Splits cleaned text into smaller, manageable chunks.
save_to_pinecone:

Saves chunks with metadata to Pinecone.
Example Workflow
Extract text from a PDF:

python
Copy code
raw_text = extract_text_from_pdf("MSME_Schemes.pdf", start_page=0, end_page=10)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text, chunk_size=1200)
Save chunks to Pinecone:

python
Copy code
metadata = {"doc_id": "1", "author": "AuthorName", "topic": "Loan Schemes"}
save_to_pinecone(chunks, metadata)
Retrieve relevant context:

python
Copy code
rag_output = retreival.retrieve_from_pinecone("What loan schemes are available for farmers?", top_k=3)
Generate a chatbot response:

python
Copy code
preprompt = prepare_preprompt(rag_output, "Provide loan schemes for farmers.")
response = llm.invoke(preprompt)
Future Enhancements
Add multilingual support for non-English users.
Integrate advanced embeddings like OpenAI's GPT-4 or SentenceTransformers.
Enable user authentication for a personalized experience.
Expand the chatbot to include other financial services.
Contributors
Vishwa: Lead Developer, PDF Knowledge Base Integration
Collaborators: Loan Scheme Analysis and UI Enhancements
License
This project is licensed under the MIT License. See the LICENSE file for more details.