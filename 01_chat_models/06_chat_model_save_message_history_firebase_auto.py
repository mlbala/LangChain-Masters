from dotenv import load_dotenv
from google.auth import compute_engine
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid
from datetime import datetime

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""


# Load env file
load_dotenv()

# Set up Firebase Firestore
# Access the variables
PROJECT_ID = os.getenv("PROJECT_ID")
# SESSION_ID = os.getenv("SESSION_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# Initilize Firestore Client
print("Initilizing Firestore Client ... ")
client = firestore.Client(project=PROJECT_ID)

# Ask user whether to create a new session or continue with the last one
# Function to save session metadata
def save_session_metadata(session_id):
    session_metadata = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
    }
    client.collection("sessions").document(session_id).set(session_metadata)
    print(f"Session metadata saved for session: {session_id}")

# Function to get the last session
def get_last_session():
    sessions = client.collection("sessions").order_by("start_time", direction=firestore.Query.DESCENDING).limit(1).stream()
    for session in sessions:
        return session.id
    return None  # No session found

# Ask user whether to create a new session or continue with the last one
def get_session_id():
    last_session_id = get_last_session()

    if last_session_id:
        print(f"Last session detected: {last_session_id}")
        choice = input("Do you want to continue with the last session? (yes/no): ").strip().lower()

        if choice == "yes":
            return last_session_id
        else:
            print("Creating a new session...")
            new_session_id = str(uuid.uuid4())
            save_session_metadata(new_session_id)
            return new_session_id
    else:
        print("No existing sessions found. Creating a new session...")
        new_session_id = str(uuid.uuid4())
        save_session_metadata(new_session_id)
        return new_session_id


# Function to print the last 5 messages
def print_last_five_messages(chat_history):
    print("\n--- Last 5 Messages ---")
    # Get the last 5 messages
    recent_messages = chat_history.messages[-5:]
    for message in recent_messages:
        if isinstance(message, HumanMessage):  # Human message
            print(f"👤 User: {message.content}")
        elif isinstance(message, AIMessage):  # AI message
            print(f"🤖 AI: {message.content}")
    # print("--- End of Chat History ---\n")

# Get the session ID
SESSION_ID = get_session_id()
print(f"Using session: {SESSION_ID}")


# Initilizing Firestore Chat Message History
print("Initilizing Firestore Chat Message History ...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("chat History Initilized.")
# print("Current Chat History.", chat_history.messages)
print("Current Chat History:")
print_last_five_messages(chat_history)


# Initilize Chat Model
model = ChatOpenAI(model="gpt-4o-mini")

# Function to display the last 5 messages dynamically
def display_dynamic_messages(last_five_messages):
    print("\033c", end="")  # Clear the terminal
    print("\n--- Last 5 Messages ---")
    for message in last_five_messages:
        if isinstance(message, HumanMessage):  # Human message
            print(f"👤 User: {message.content}")
        elif isinstance(message, AIMessage):  # AI message
            print(f"🤖 AI: {message.content}")
    # print("--- End of Chat History ---\n")


# while True:
#     human_input = input("User: ")
#     if human_input.lower() == "exit":
#         break

#     chat_history.add_user_message(human_input)

#     ai_response = model.invoke(chat_history.messages)
#     chat_history.add_ai_message(ai_response.content)

#     # recent_messages = chat_history.messages[-5:]

#     # for message in recent_messages:
#     #     if message.type == "user":
#     #         print(f"👤 User: {message.content}")
#     #     elif message.type == "ai":
#     #         print(f"🤖 AI: {message.content}")
#     # # print("--- End of Chat History ---\n")

#     # Display AI response
#     print(f"🤖 AI: {ai_response.content}")

#     # Print the last 5 messages in chat history
#     print_last_five_messages(chat_history)

# Initialize last 5 messages as a sliding window
last_five_messages = chat_history.messages[-5:]  # Fetch initial last 5 messages if any
display_dynamic_messages(last_five_messages)

# Chat loop
while True:
    # Get raw input from the user
    human_input = input("👤 User: ")
    if human_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user input to Firestore chat history as a HumanMessage object
    user_message = HumanMessage(content=human_input)
    chat_history.add_user_message(user_message.content)

    # Append the user's message to the sliding window
    last_five_messages.append(user_message)
    if len(last_five_messages) > 5:  # Ensure only the last 5 messages are kept
        last_five_messages.pop(0)

    # Generate AI response
    ai_response = model.invoke(chat_history.messages)
    ai_message = AIMessage(content=ai_response.content)
    chat_history.add_ai_message(ai_message.content)

    # Append the AI response to the sliding window
    last_five_messages.append(ai_message)
    if len(last_five_messages) > 5:  # Ensure only the last 5 messages are kept
        last_five_messages.pop(0)
     # Dynamically display the last 5 messages
    display_dynamic_messages(last_five_messages)






    # # Generate AI response
    # ai_response = model.invoke(chat_history.messages)

    # # Add AI response to Firestore chat history as an AIMessage object
    # ai_message = AIMessage(content=ai_response.content)
    # chat_history.add_ai_message(ai_message.content)

    # # Display AI response
    # print(f"🤖 AI: {ai_response.content}")

    # # Print the last 5 messages in chat history
    # print_last_five_messages(chat_history)