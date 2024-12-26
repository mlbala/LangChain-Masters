import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

# Load environemnt variable
load_dotenv()

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Data directory: {data_dir}")
print(f"Persistent directory: {persistent_directory}")

# Ensure the data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The directory {data_dir} does not exist. Please check the path.")

# Ensure the persistent directory exists
if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)
    print(f"Created persistent directory at {persistent_directory}")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load or create the vector store
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    # Load the existing vector store
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    # Retrieve the list of documents already in the vector store
    existing_doc_ids = set()
    collection = db._collection
    if len(collection.get()["ids"]) > 0:
        existing_doc_ids = set(collection.get()["ids"])
    else:
        print("No documents found in the existing vector store.")
else:
    print("Persistent directory does not exist. Initializing new vector store...")
    existing_doc_ids = set()

# List all text files in the directory
data_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

# Identify new documents to process
new_documents = []
for data_file in data_files:
    doc_id = data_file  # Using filename as a unique identifier
    if doc_id not in existing_doc_ids:
        print(f"Found new document: {data_file}")
        file_path = os.path.join(data_dir, data_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source and ID
            doc.metadata = {"source": data_file, "doc_id": doc_id}
            new_documents.append(doc)
    else:
        print(f"Document {data_file} is already in the vector store.")

# Process new documents if any
if new_documents:
    print(f"\nProcessing {len(new_documents)} new document(s)...")

    # Split the documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(new_documents)

    # Add the new documents to the vector store
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
else:
    print("No new documents to process.")


