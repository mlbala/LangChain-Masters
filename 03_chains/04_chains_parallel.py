from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variable fro .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define prompt template
prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main feature of the product {product_name}."),
    ]
)

# Define pros analysis step

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an expert product reviewer."),
            ("human", "Given these features : {features}, list the pros of tehse features."),
        ]
    )
    return pros_template.format_prompt(features=features)

# define cons analysis steps
def analyze_cons(features):
    cons_tempalte = ChatPromptTemplate.from_messages(
        [
            ("system", "You are n expert product reviewer."),
            ("human", "Given these features : {features}, list the cons of these features."),
        ]
    )
    return cons_tempalte.format_prompt(features=features)

# Combine pros and cons into a final review 

def combine_pros_cons(pros, cons):
    return f"Pros: \n{pros}\n\nCons:\n{cons}"

# Simplify branches with LCEL
pros_branch_chain = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()

cons_branch_chain = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()

## Create the combined chain using LangChain Expression Language LCEL

chain = (
    prompt_template 
    | model 
    | StrOutputParser()
    | RunnableParallel(branches = {"pros" : pros_branch_chain, "cons" : cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
    # | RunnableLambda(lambda x: print("final output", x) or combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# run the chain
result = chain.invoke({"product_name" : "MacBook Air M1"})

# Output
print(result)