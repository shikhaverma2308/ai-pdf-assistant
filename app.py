import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("🤖 AI PDF Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully ✅")

    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)

    st.subheader("Ask questions from your PDF 👇")

    query = st.text_input("Type your question:")

    if query:
        with st.spinner("Thinking... 🤔"):
            docs = db.similarity_search(query)
            context = docs[0].page_content

            generator = pipeline("text-generation", model="google/flan-t5-base")

            prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}\nAnswer:"

            response = generator(prompt, max_length=200)

            st.write("### 🤖 Answer:")
            st.write(response[0]["generated_text"])