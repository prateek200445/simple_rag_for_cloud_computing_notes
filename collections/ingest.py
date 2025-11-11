from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

pdf_folder = "data_pdfs"
# Load all PDFs
all_documents = []
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file_name)
        print(f"Loading: {file_name}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

print(f"Total PDFs loaded: {len(all_documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250
)
texts = text_splitter.split_documents(all_documents)
print(f"Total text chunks created: {len(texts)}")

# Load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("✅ Embeddings loaded")

# Connect to Qdrant and push data
url = "http://localhost:6333"
collection_name = "pdf_db"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name
)

print(" Qdrant loaded and data ingested successfully")
