from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy.orm import Session
from sqlalchemy import text
from models import engine, DocumentChunk

print("Loading Embedding Model (The Net)...")
embed_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

print("Loading Cross-Encoder (The Scalpel)...")
# This will download the reranker weights on first run
reranker = CrossEncoder('BAAI/bge-reranker-base')

def enterprise_search(user_query):
    print(f"\n🔍 QUERY: {user_query}")
    print("="*60)
    
    with Session(engine) as session:
        # --- STAGE 1: HYBRID RETRIEVAL ---
        
        # 1A. Vector Search (Top 10 Conceptual Matches)
        query_vector = embed_model.encode(user_query).tolist()
        vector_results = session.query(DocumentChunk)\
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))\
            .limit(10).all()
            
        # 1B. BM25 Keyword Search (Top 10 Exact Matches using Postgres Native Text Search)
        keyword_sql = text("""
            SELECT id 
            FROM document_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
            LIMIT 10;
        """)
        keyword_ids = [row[0] for row in session.execute(keyword_sql, {"query": user_query}).fetchall()]
        keyword_results = session.query(DocumentChunk).filter(DocumentChunk.id.in_(keyword_ids)).all()
        
        # Combine and deduplicate
        unique_chunks = {chunk.id: chunk for chunk in (vector_results + keyword_results)}
        candidate_chunks = list(unique_chunks.values())
        
        if not candidate_chunks:
            print("❌ No relevant documents found.")
            return

        print(f"🎣 Stage 1 Complete: Retrieved {len(candidate_chunks)} candidates via Hybrid Search.")

        # --- STAGE 2: CROSS-ENCODER RE-RANKING ---
        
        # Prepare the pairs for the scalpel: [[Query, Chunk1], [Query, Chunk2], ...]
        pairs = [[user_query, chunk.content] for chunk in candidate_chunks]
        
        # The reranker outputs a raw logit score. Higher is better.
        scores = reranker.predict(pairs)
        
        # Attach scores to chunks and sort them descending
        scored_results = list(zip(candidate_chunks, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # --- THE OUTPUT ---
        # We only care about the absolute best result now.
        best_chunk, best_score = scored_results[0]
        
        print(f"🎯 Stage 2 Complete: Re-ranked for absolute relevance.")
        print(f"\n🏆 TOP RESULT (Score: {best_score:.2f})")
        print(f"Source: {best_chunk.metadata_['source']} | Page {best_chunk.metadata_['page']}")
        print(f"Text: {best_chunk.content.strip()}\n")

if __name__ == "__main__":
    # Test 1: Conceptual search
    enterprise_search("What happens if I use an unapproved messaging app?")
    
    # Test 2: Specific keyword that might confuse a pure vector search
    enterprise_search("Are vendor gifts over $50 allowed?")

    enterprise_search("What is vertical scaling")

    enterprise_search("How does the token bucket algorithm control the rate of incoming requests, and what happens to extra tokens when the bucket is full?")