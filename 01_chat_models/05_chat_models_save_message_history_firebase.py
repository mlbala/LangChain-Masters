
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
import os


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

load_dotenv()


# Set up Firebase Firestore
# Access the variables
PROJECT_ID = os.getenv("PROJECT_ID")
SESSION_ID = os.getenv("SESSION_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

#Initilize firestore Client

print("Initilizing Firestore Cfirestore Client ...")
client = firestore.Client(project=PROJECT_ID)

# Initilize Firestore Chat Message history
print("Initilizing Firestore Chat Message History ...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Chat History Initilized.")
print("current Chat History: ", chat_history.messages)

# Initilize Chat Model

model = ChatOpenAI(model = "gpt-4o-mini")

print("Start chat with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User : ")
    if human_input.lower() == 'exit':
        print("Good Bye!")
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
     # Print conversation in human-readable format
    # print("\n--- Conversation ---")
    # print(f"User: {human_input}")
    # print(f"AI: {ai_response.content}")
    # print("--------------------\n")
    # Print output like ChatGPT
    print(f"\nAI : {ai_response.content}\n")