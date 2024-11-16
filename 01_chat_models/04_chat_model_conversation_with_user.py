from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model

model = ChatOpenAI(model = "gpt-4o-mini")

#use a list to store messages
chat_history = [] 

# Set an initial system message 

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message) #add system message

# Chat look
while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) # add user query

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) # AI message
    
    print(f"AI : {response}")

    # Optional: Truncate chat history if it grows too large
    max_messages = 20  # Keep the last 20 messages + system message
    if len(chat_history) > max_messages:
        chat_history = [system_message] + chat_history[-(max_messages - 1):]

print("----- Message History -----")
# print(chat_history)

for message in chat_history:
    if isinstance(message, HumanMessage):
        print(f"User: {message.content}")
    elif isinstance(message, SystemMessage):
        print(f"System: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")