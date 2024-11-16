from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

#Setup Environment Variable and messages
# load_dotenv()

message = [
    SystemMessage(content="Solve the following math probles"),
    HumanMessage(content="What is 1024 divided by 8?"),
]

## ----------- LangChain OpenAI Chat Model Example -----------------------#

#create a ChatOpenAI model

model = ChatOpenAI(model = "gpt-4o-mini")

#Invoke the model with message

result = model.invoke(message)
print(f"Answer from Open AI : {result.content}")

# # --------------GOOGLE Chat Models Examples ----------------- #

# # https://console.cloud.google.com/gen-app-builder/engines
# # https://ai.google.dev/gemini-api/docs/models/gemini

model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

result = model.invoke(message)
print(f"Answer fro Google model : {result.content}")


## ------------------------------- Anthropic model ------------------#


# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview

# model = ChatAnthropic(model = "claude-3-opus-20240229")

# result = model.invoke(message)
# print(f"Answer from Anthropic : {result.content}")