from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

#Load environment variables

load_dotenv()

#Create a chatOpenAI model

model = ChatOpenAI(model = "gpt-4o-mini")

# SystemMessage
#   Message for priming AI behaviour, usually passed in as the first of a sequence of Input messages.
# HumanMessage:
#   Message from a huma to the AI model.

message = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 1024 divided by 8?"),
    ] 

#Invoke the model with messages

result = model.invoke(message)
print(f"Answer from AI :  {result.content}")

