from langchain_ollama import OllamaLLM
import retreival

# Initialize LLaMA
llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434")

# query = "Enter you query"

# rag_output = retreival.retrieve_from_pinecone(query,4)

# # Initialize chat history
# chat_history = []

# Function to prepare the pre-prompt
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

    If a user provides additional criteria, incorporate it into your response to refine your advice.
    """
    return preprompt


# # Preprompt instructions
# instructions = "Provide suitable loan schemes for users based on their criteria."

# # Prepare the pre-prompt
# preprompt = prepare_preprompt(rag_output, instructions)

# # Interactive chat loop
# print("Chat with LLaMA (type 'exit' to quit):")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("Exiting chat. Goodbye!")
#         break

#     # Combine pre-prompt, chat history, and user input
#     conversation = (
#         preprompt
#         + "\n\n"  # Separate the pre-prompt from the conversation
#         + "\n".join([f"User: {h['user']}\nLLaMA: {h['llm']}" for h in chat_history])
#         + f"\nUser: {user_input}\nLLaMA:"
#     )

#     # Get LLM response
#     response = llm.invoke(conversation)
#     print(f"LLaMA: {response}")

#     # Add to chat history
#     chat_history.append({"user": user_input, "llm": response})
