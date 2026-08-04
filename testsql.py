import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

# The exact connection string mapping to your Docker container
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/vectordb"

def verify_infrastructure():
    print("🔌 Attempting to connect to PostgreSQL...")
    
    # Initialize the engine
    engine = create_engine(DATABASE_URL)
    
    try:
        with Session(engine) as session:
            # --- TEST 1: Basic Connection ---
            session.execute(text("SELECT 1"))
            print("✅ TEST 1 PASSED: Successfully connected to PostgreSQL.")
            
            # --- TEST 2: Verify PGVector Extension ---
            # Interrogate Postgres's internal extension registry
            result = session.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
            )).fetchone()
            
            if result:
                print(f"✅ TEST 2 PASSED: PGVector is ACTIVE (Version {result[0]}).")
            else:
                print("❌ TEST 2 FAILED: PGVector is not active in this database.")
                print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")
                sys.exit(1)
                
            # --- TEST 3: End-to-End Vector Math ---
            print("\n🧪 Running End-to-End Vector Engine Test...")
            
            # Create a temporary table that is automatically destroyed when the script ends
            session.execute(text("""
                CREATE TEMP TABLE vector_diagnostic (
                    id SERIAL PRIMARY KEY,
                    embedding VECTOR(3)
                );
            """))
            
            # Insert two test vectors
            session.execute(text("""
                INSERT INTO vector_diagnostic (embedding) 
                VALUES ('[1, 2, 3]'), ('[4, 5, 6]');
            """))
            
            # Query the nearest neighbor to [1, 2, 4] using L2 distance (<->)
            closest_match = session.execute(text("""
                SELECT embedding, embedding <-> '[1, 2, 4]' AS distance 
                FROM vector_diagnostic 
                ORDER BY distance LIMIT 1;
            """)).fetchone()
            
            if closest_match:
                print(f"✅ TEST 3 PASSED: Vector engine successfully calculated distance.")
                print(f"   🎯 Closest vector to [1, 2, 4] is {closest_match[0]} (Distance: {closest_match[1]:.4f})")
            
            print("\n🚀 ALL SYSTEMS GO. Your database layer is ready for ingestion.")
            
    except OperationalError as e:
        print("\n❌ CRITICAL FAILURE: Could not connect to the database.")
        print("   1. Did you run the Docker command?")
        print("   2. Is Docker Desktop actually running in the background?")
        print(f"\n   Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_infrastructure()