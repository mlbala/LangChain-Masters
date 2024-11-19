from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI


# Load environment variable from .env
load_dotenv()

# Create a Chatopenai model
model = ChatOpenAI(model="gpt-4o-mini")

# Define prompt templates 
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additonal processing steps using RunnableLambda

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"word count : {len(x.split())}\n{x}")

# Create teh combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic" : "lawyers", "joke_count" : 3})

# Output
print(result)



# ## For multiple topic and joke count 
# # Topics and joke counts
# topics_and_counts = [
#     {"topic": "lawyers", "joke_count": 3},
#     {"topic": "software engineers", "joke_count": 5},
# ]

# # Collect jokes for each topic
# results = {}
# for entry in topics_and_counts:
#     topic = entry["topic"]
#     joke_count = entry["joke_count"]
#     result = chain.invoke({"topic": topic, "joke_count": joke_count})
#     results[topic] = result

# # Output the jokes for each topic
# for topic, jokes in results.items():
#     print(f"Jokes about {topic}:\n{jokes}\n")