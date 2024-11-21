import os
# from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environemnt variable
# load_dotenv()

# Define the directory containing the text file and the persistent directoy
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data","odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exist
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initilizing vector store ...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" The File {file_path} does not exist. Please check the path")

    # Read the text content fro the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n --- Documents Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n --- Creating Embeddings ---")
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") 
    print("\n --- Fineshed creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n --- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n --- Fineshed creating vector store --- ")

else:
    print("Vector store already exists, No need to initilize")