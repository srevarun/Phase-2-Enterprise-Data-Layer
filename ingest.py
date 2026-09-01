import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from models import engine, DocumentChunk  # Importing your DB engine from earlier
import uuid

# 1. Load the local embedding model (generates 768-dimensional vectors)
# This will download the model weights the first time you run it.
print("Loading embedding model...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# 2. Configure the Chunking Strategy
# 1000 characters per chunk, with 200 characters of overlap to prevent cutting sentences in half.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def process_pdf(file_path):
    print(f"Extracting text from {file_path}...")
    doc = fitz.open(file_path)
    
    chunks_to_insert = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        if not text.strip():
            continue
            
        # Split the page text into logical chunks
        page_chunks = text_splitter.split_text(text)
        
        for chunk_text in page_chunks:
            # Generate the 768-dimension vector
            embedding = model.encode(chunk_text).tolist()
            
            # Package it into the SQLAlchemy model
            db_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content=chunk_text,
                metadata_={"source": file_path, "page": page_num + 1},
                embedding=embedding
            )
            chunks_to_insert.append(db_chunk)
            
    return chunks_to_insert

def seed_database(chunks):
    print(f"Inserting {len(chunks)} chunks into PostgreSQL...")
    with Session(engine) as session:
        session.bulk_save_objects(chunks)
        session.commit()
    print("Ingestion complete. Database is armed.")

if __name__ == "__main__":
    # Ensure you have a PDF named compliance_manual.pdf in this folder
    chunks = process_pdf("System_Design.pdf")
    seed_database(chunks)