from pypdf import PdfReader
from langchain_text_splitter import RecursiveCharacterTextSplitter

# Load PDF
reader = PdfReader("shikhaverma.resume.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))
print("\nFirst chunk:\n", chunks[0])