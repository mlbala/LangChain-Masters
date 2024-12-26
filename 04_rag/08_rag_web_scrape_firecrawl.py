import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")

def create_vector_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    # Define the Firecrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise ValueError("Firecrawl API key not found in environment variables.")
    
    # Step  1: Crawl the website using FireCrawlLoader
    print("\n--- Crawling the website ---")
    loader = FireCrawlLoader(api_key=api_key,  url="https://apple.com", mode="scrape")

    docs = loader.load()
    print("\n--- Finished crawling the website ---")

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Step 2: Split the scraped content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Display information  about the split documents
    print("\n--- Documents Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Step 3: Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 4: Create the vector store and persist it
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store in {persistent_directory} ---")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(f"Vector store already exists at {persistent_directory}. Skipping vector store creation.")

# Load the vector store with the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Step 5: Query the vector store
def query_vector_store(query):
    """Query the vector store and return the results."""
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    # Retrieve relevant documents based on the query
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Define the user's question
query = "Apple Intellience?"

# Query the vector store with the user's question
query_vector_store(query)

