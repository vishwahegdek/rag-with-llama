import streamlit as st
from langchain_ollama import OllamaLLM

import json
import time
import retreival
import query_bot

llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434")

def prepare_preprompt(rag_output, instructions):
    """
    Combine RAG output and instructions into a pre-prompt.
    """
    context = "\n".join([f"- {chunk['metadata']['text']}" for chunk in rag_output["matches"]])
    preprompt = f"""
    You are an intelligent assistant specializing in providing advice on loan schemes based on specific criteria.

    Below is some contextual information retrieved from a knowledge base (RAG output):
    {context}

    Your task:
    - Use the above contextual information to identify suitable loan schemes.
    - Respond to user queries by explaining relevant schemes clearly and concisely.
    - Do not go off topic from the context provided.
    - Stick to the context paragraph provided
    If a user provides additional criteria, incorporate it into your response to refine your advice.
    """
    return preprompt

st.title("CashFlow Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chats" not in st.session_state:
    st.session_state.chats = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "next_clicked" not in st.session_state:
    st.session_state.next_clicked = False  # To handle single click per question
if "chat_on" not in st.session_state:
    st.session_state.chat_on = False
if "processing" not in st.session_state:
    st.session_state.processing = False


scheme_questions = [
    {
        "question": "What is your annual income?",
        "type": "select",
        "options": ["Less than ₹1,00,000", "₹1,00,000–₹3,00,000", "₹3,00,000–₹5,00,000", "₹5,00,000–₹7,50,000", "₹7,50,000–₹10,00,000", "More than ₹10,00,000"],
    },
    {
        "question": "What is your occupation?",
        "type": "select",
        "options": ["Construction Worker", "Unorganized Worker", "Farmer", "Student", "Artists", "Artisans", "Spinners & Weavers", "Fishermen", "Organized Worker", "Ex Servicemen", "Journalist", "Safai Karamchari", "Sportsperson", "Coir Worker", "Teacher / Faculty", "Health Worker", "Khadi Artisan", "Street Vendor"],
    },
]

# Function for typing animation
def typing_animation(text, role="assistant"):
    """Simulates typing animation for the assistant."""
    with st.chat_message(role):
        placeholder = st.empty()
        typed_text = ""
        for char in text:
            typed_text += char
            placeholder.markdown(typed_text)
            time.sleep(0.02)  # Adjust typing speed
        placeholder.markdown(typed_text)  # Finalize the text

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Current question handling
if st.session_state.current_question < len(scheme_questions):
    current_q = scheme_questions[st.session_state.current_question]

    # Show the question
    typing_animation(current_q["question"])

    # Response input
    response = None
    if current_q["type"] == "select":
        response = st.selectbox(
            "Choose an option:",
            ["Select an option..."] + current_q["options"],  # Add placeholder option
            key=f"q{st.session_state.current_question}",
        )
    elif current_q["type"] == "number":
        response = st.number_input(
            "Enter a number:", step=1, key=f"q{st.session_state.current_question}"
        )
    elif current_q["type"] == "text":
        response = st.text_input(
            "Your answer:", key=f"q{st.session_state.current_question}"
        )

    # Handle the "Next" button
    if st.button("Next", key="next_button"):
        if not st.session_state.next_clicked:  # Ensure button works once per question
        # Validate response
            if current_q["type"] == "select" and response == "Select an option...":
                st.warning("Please select a valid option before proceeding.")
            elif response not in [None, ""]:  # Valid response
                st.session_state.next_clicked = True  # Prevent multiple clicks

                # Save the response and proceed
                st.session_state.messages.append(
                    {"role": "user", "content": str(response)}
                )
                st.session_state.responses.append(response)

                # Advance to the next question
                st.session_state.current_question += 1
                st.rerun()

    # Reset the "Next" button state after moving to the next question
    if st.session_state.next_clicked and st.session_state.current_question > 0:
        st.session_state.next_clicked = False

# When all questions are answered
elif st.session_state.current_question == len(scheme_questions):
    typing_animation("You have completed the questionnaire!")

    if st.button("Submit"):
        # Combine questions and responses into a structured JSON format
        structured_responses = [
            {"question": scheme_questions[i]["question"], "response": st.session_state.responses[i]}
            for i in range(len(st.session_state.responses))
        ]

        # Save the structured responses to a JSON file
        with open("responses.json", "w") as f:
            json.dump(structured_responses, f, indent=4)

        typing_animation("Your responses have been successfully submitted and saved as a JSON file!")

    if st.button("Retake"):
        # Reset the session state
        st.session_state.messages = []
        st.session_state.responses = []
        st.session_state.current_question = 0

    if st.button("Chat with us"):
        st.session_state.chat_on = True



    if st.session_state.chat_on:  
        # Chatbot interaction after the questionnaire
        st.divider()  # Adds a visual separation between the questionnaire and chatbot
        st.subheader("Chat with the Assistant")
        # Display chat history
        for chat in st.session_state.chats:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # Chatbot Input
        user_input = st.text_input("Type your message:", key="chat_input")
        
        # Disable the "Send" button while processing
        send_button_disabled = st.session_state.processing or not user_input.strip()
        if st.button("Send", key="chat_send", disabled=send_button_disabled):
            # Set processing to True
            st.session_state.processing = True

            # Save user's message
            st.session_state.chats.append({"role": "user", "content": user_input})

            # Retrieve relevant context using RAG pipeline
            rag_output = retreival.retrieve_from_pinecone(user_input, 4)

            # Prepare the pre-prompt
            instructions = "Provide suitable loan schemes for users based on their criteria."
            preprompt = query_bot.prepare_preprompt(rag_output, instructions)

            # Combine pre-prompt, chat history, and user input
            conversation = (
                preprompt
                + "\n\n"
                + "\n".join(
                    [
                        f"User: {chat['content']}\nLLaMA: {chat.get('response', '')}"
                        for chat in st.session_state.chats
                        if chat["role"] == "user"
                    ]
                )
                + f"\nUser: {user_input}\nLLaMA:"
            )

            # Get response from LLaMA
            response = llm.invoke(conversation)

            # Add LLaMA's response to the chat history
            st.session_state.chats.append({"role": "assistant", "content": response})

            # Reset processing to False after completion
            st.session_state.processing = False

        

