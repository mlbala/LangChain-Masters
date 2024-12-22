import os
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

# Load environemnt variable
load_dotenv()
# define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persisitent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the exisiting vector store with the embedding funciton
db = Chroma(persist_directory=persisitent_directory, embedding_function=embeddings)

# define the user's query
query = "Who is Odysseus' wife?"

# Ratetrieve relevant documents based on the query
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.9},)
relevant_docs = retriever.invoke(query)

# Display the relevant result with metadata
print("\n --- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f" Document {i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source : {doc.metadata.get('source','Unknown')}\n")