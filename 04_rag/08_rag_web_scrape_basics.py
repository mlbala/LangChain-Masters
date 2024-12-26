import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_directory = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_directory, "chroma_db_apple")

# Step 1: Scrape the content from apple.com using WebBaseLoader
# WebBaseLoader loads web pages and extracts their content
urls = ["https://www.apple.com/"]

# Create a loader for web content
loader = WebBaseLoader(urls)
documents = loader.load()

# Step 2: Split the scraped content into chunks
# CharacterTextSplitter splits the text into chunks of a specified size
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n --- Documents Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Step 3: Create embeddings
# OpenAIEmbeddings turns text into numerical vectors that capture semantic meaning
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 4: Create and persist the vector store with the embeddings
# Chroma stores the embeddings in a database for efficient retrieval
if not os.path.exists(persistent_directory):
    print(f"\n --- Creating vector store in {persistent_directory}---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"\n --- Fineshed creating vector store in {persistent_directory} --- ")
else:
    print(f"\n --- Vector store already exists in {persistent_directory}, No need to initilize ---")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Step 5: Query the vector store
# create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the user's question
query = "What products does Apple offer?"

# Retrieve relevant documents based on the query
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")