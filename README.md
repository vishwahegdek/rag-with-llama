# CashFlow Chatbot

## Overview

CashFlow Chatbot is a dynamic web application built using **Streamlit** that interacts with users to provide advice on loan schemes based on their specific criteria. The system employs **Retrieval-Augmented Generation (RAG)**, **LLM (LLaMA3.1:8b)**, and **Pinecone** for intelligent context retrieval and response generation. It also includes the ability to preprocess and analyze PDF documents for knowledge base creation, embedding generation, and indexing.

---

## Features

- **Interactive Questionnaire**: 
  - Users answer a series of questions to provide input for loan scheme recommendations.
  - Includes validations for responses and session persistence.

- **Real-time Chatbot**:
  - Users can chat with an assistant after completing the questionnaire.
  - The assistant uses RAG to retrieve relevant context and LLM to generate concise, meaningful responses.

- **PDF Knowledge Base Extraction**:
  - Extracts and preprocesses text from PDF files.
  - Splits text into chunks for efficient indexing and retrieval.

- **Knowledge Base Indexing**:
  - Uses **Pinecone** for vector-based indexing and similarity search.
  - Embeddings generated with **BERT** are stored in Pinecone for quick access.

- **Typing Animation**:
  - Provides a realistic, engaging chatbot interaction.

---

## Tech Stack

### Backend Components:
- **Streamlit**: For the interactive user interface.
- **LangChain & LLaMA**: For large language model inference and RAG pipeline.
- **PyPDF2**: For PDF text extraction.
- **NLTK**: For text preprocessing and sentence tokenization.
- **Transformers (HuggingFace)**: For embedding generation using the **BERT** model.
- **Pinecone**: For vector database management and efficient retrieval.

### Frontend Features:
- Responsive and intuitive interface.
- Dynamic input handling for questionnaire responses.
- Chatbot interaction with real-time user feedback.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or later.
- Basic understanding of NLP and Streamlit.
- Access to a Pinecone API key and LLaMA model.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/cashflow-chatbot.git
   cd cashflow-chatbot
