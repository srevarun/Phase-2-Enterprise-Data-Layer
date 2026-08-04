from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/vectordb"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    embedding = Column(Vector(768))  # 768 dimensions for BGE-base model

def init_db():
    print("🏗️  Building database schema...")
    # This automatically generates and runs the CREATE TABLE SQL
    Base.metadata.create_all(engine)
    print("✅ Schema built. Tables are ready.")

if __name__ == "__main__":
    init_db()