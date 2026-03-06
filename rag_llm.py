from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# Load PDF
reader = PdfReader("shikhaverma.resume.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Store in FAISS
db = FAISS.from_texts(chunks, embeddings)

# Load LLM (FREE 🔥)
generator = pipeline("text-generation", model="gpt2")

# Ask question
query = input("Ask your question: ")

docs = db.similarity_search(query)
context = docs[0].page_content

# Generate answer
prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}\nAnswer:"

response = generator(prompt, max_length=200, num_return_sequences=1)

print("\n🤖 AI Answer:\n")
print(response[0]["generated_text"])