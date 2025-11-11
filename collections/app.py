from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
import google.generativeai as genai
import sys
import os

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
collection_name = "pdf_db"

# Initialize Qdrant client
client = QdrantClient(url=url, prefer_grpc=False)
print(client)
print("✅ Qdrant client initialized ##################")

# Connect to the collection
db = Qdrant(
    collection_name=collection_name,
    embeddings=embeddings,
    client=client
)

print(db)
print("✅ Qdrant vector store initialized ##################")

# Query
print("\n RAG Chatbot Ready! Type 'exit' to quit.\n")
while True:
    query = input(" Enter your question: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting the chatbot. Goodbye!")
        break


    docs = db.similarity_search_with_score(query, k=5)

    if len(docs) == 0 or docs[0][1] < 0.7:
        print("No relevant documents found.")
        continue
    

    for doc, score in docs:
        print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

# Build prompt for LLM
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    print("\nFinal Prompt:\n")
    print(prompt)


    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    print("Response:\n", response.text)

    sources = [doc.metadata.get("source", None) for doc, _score in docs]
    formatted_response = f"Response: {response.text}\nSources: {sources}"
    print(formatted_response)

    for i, (doc, score) in enumerate(docs, 1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown')} (score={score:.2f})")

    print("\n" + "-"*60 + "\n")
