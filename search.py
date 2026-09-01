from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from models import engine, DocumentChunk

print("Loading model...")
# You MUST use the exact same model you used to ingest the data
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

def semantic_search(user_query):
    # 1. Convert the user's text into a 768-dimension vector
    query_vector = model.encode(user_query).tolist()
    
    with Session(engine) as session:
        # 2. The Math: Calculate Cosine Distance between the query vector and every chunk in the DB.
        # Lower distance = closer meaning.
        distance_expr = DocumentChunk.embedding.cosine_distance(query_vector).label('distance')
        
        # 3. Retrieve the top 2 closest chunks
        results = session.query(DocumentChunk, distance_expr)\
                         .order_by(distance_expr)\
                         .limit(2)\
                         .all()
                         
        print(f"\n🔍 QUERY: {user_query}\n" + "="*50)
        for chunk, distance in results:
            # Convert the mathematical distance into a readable similarity percentage
            similarity = (1 - distance) * 100
            print(f"[{similarity:.1f}% Match] Source: {chunk.metadata_['source']} (Page {chunk.metadata_['page']})")
            print(f"Text: {chunk.content.strip()}\n" + "-"*50)

if __name__ == "__main__":
    # Test Query 1: Direct keyword match
    semantic_search("What is the penalty for using WhatsApp?")
    
    # Test Query 2: Pure conceptual match (Notice how the word "bribe" isn't in the PDF, but "vendor gift" is)
    semantic_search("Am I allowed to accept a bribe from a supplier?")

    semantic_search("What is the theorem used to imply that 80% of the data is contained in 20% of the population?")

    semantic_search("What are the Advantages of database replication")

    