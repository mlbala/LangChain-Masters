from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI


# Load the Environemnt variable from .env

load_dotenv()

# Create a chatOpenAI model

model = ChatOpenAI(model="gpt-4o-mini")

#Define prompt templates

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create Induvidual runnables (steps in the chain)

format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x : x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic":"lawyers", "joke_count": 5})

#output
print(response)