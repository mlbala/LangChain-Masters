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

# # Run the chain
# response = chain.invoke({"topic":"lawyers", "joke_count": 5})

# #output
# print(response)


# # For multiple topic
# # Topics to generate joeks for
# topics = ["lawyers","Software engineer","HR"]

# # collect jokes for each topic
# all_jokes  = {}
# for topic in topics:
#     response = chain.invoke({"topic": topic, "joke_count": 3})
#     all_jokes[topic] = response

# # Output the jokes

# for topic, jokes in all_jokes.items():
#     print(f"Jokes about {topic} : \n{jokes}\n")

# For Dynamically specify the number of jokes for each topic

# Topics and jokes count
topics_and_counts = [
    {"topic" : "lawyers","joke_count": 3},
    {"topic" : "software engineer", "joke_count":2},
    {"topic" : "HR", "joke_count": 3},
]

# Collect jokes  for each topic
all_jokes = {}
for entry in topics_and_counts:
    topic = entry["topic"]
    joke_count = entry["joke_count"]
    response = chain.invoke({"topic": topic, "joke_count": joke_count})
    all_jokes[topic] = response

# Output the jokes

for topic, jokes in all_jokes.items():
    print(f"Jokes about {topic} : {jokes}")
