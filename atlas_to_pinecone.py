"""
Upload BioBERT Embeddings from MongoDB Atlas to Pinecone
- Reads embedded chunks from Atlas
- Uploads to Pinecone with metadata
- Tracks sync status
"""

import pymongo
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from tqdm import tqdm
import time
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medicrew")
CHUNKS_COLLECTION = "paper_chunks"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "medical-papers-biobert"
EMBEDDING_DIMENSION = 768  # BioBERT dimension
METRIC = "cosine"

# Processing
UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 500  # Fetch in batches to avoid cursor timeout
RATE_LIMIT_DELAY = 0.5


# ============================================================
# SETUP
# ============================================================

def validate_config():
    """Validate environment variables"""
    errors = []
    
    if not MONGO_URI:
        errors.append("MONGO_URI not set in .env")
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not set in .env")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        exit(1)


def setup_mongodb():
    """Connect to MongoDB Atlas"""
    print("Connecting to MongoDB Atlas...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        chunks_collection = db[CHUNKS_COLLECTION]
        client.server_info()
        print(f"Connected: {CHUNKS_COLLECTION}")
        return chunks_collection
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        exit(1)


def setup_pinecone():
    """Initialize Pinecone"""
    print("\nInitializing Pinecone...")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_NAME not in existing_indexes:
            print(f"Creating new index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            print(f"Index created")
            time.sleep(2)
        else:
            print(f"Using existing index: {INDEX_NAME}")
        
        index = pc.Index(INDEX_NAME)
        
        # Show stats
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        print(f"Current vectors: {vector_count:,}")
        
        return index
        
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
        exit(1)


# ============================================================
# UPLOAD
# ============================================================

def prepare_pinecone_vector(chunk: Dict) -> Dict:
    """Prepare chunk for Pinecone upload"""
    
    vector_id = f"{chunk['pmid']}_chunk_{chunk['chunk_index']}"
    
    metadata = {
        'pmid': chunk['pmid'],
        'chunk_index': chunk['chunk_index'],
        'chunk_type': chunk['chunk_type'],
        'section': chunk.get('section', '')[:100],
        'title': chunk.get('title', '')[:500],
        'journal': chunk.get('journal', '')[:200],
        'year': chunk.get('year'),
        'domain': chunk.get('domain', ''),
        'token_count': chunk.get('token_count'),
        'pubmed_url': chunk.get('pubmed_url', ''),
        'text_preview': chunk['text'][:200]
    }
    
    # Remove None values
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return {
        'id': vector_id,
        'values': chunk['embedding'],
        'metadata': metadata
    }


def upload_to_pinecone(chunks_collection, pinecone_index):
    """Upload embeddings to Pinecone with batch fetching"""
    
    # Count total to upload
    total = chunks_collection.count_documents({
        'embedded': True,
        'synced_to_pinecone': False
    })
    
    if total == 0:
        print("\nAll chunks already synced to Pinecone")
        return
    
    print(f"\nUploading {total:,} vectors to Pinecone...")
    print(f"Index: {INDEX_NAME}")
    print(f"Batch size: {UPSERT_BATCH_SIZE}")
    print()
    
    uploaded = 0
    
    with tqdm(total=total, desc="Uploading", unit="vectors") as pbar:
        while uploaded < total:
            # Fetch a batch of chunks (avoid cursor timeout)
            chunks = list(chunks_collection.find(
                {
                    'embedded': True,
                    'synced_to_pinecone': False
                },
                limit=FETCH_BATCH_SIZE
            ))
            
            if not chunks:
                break
            
            # Process in upsert batches
            for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
                batch = chunks[i:i+UPSERT_BATCH_SIZE]
                
                # Prepare vectors
                vectors = [prepare_pinecone_vector(chunk) for chunk in batch]
                
                try:
                    # Upload to Pinecone
                    pinecone_index.upsert(vectors=vectors)
                    
                    # Mark as synced in MongoDB
                    chunk_ids = [chunk['_id'] for chunk in batch]
                    chunks_collection.update_many(
                        {'_id': {'$in': chunk_ids}},
                        {
                            '$set': {
                                'synced_to_pinecone': True,
                                'synced_at': datetime.now(timezone.utc)
                            }
                        }
                    )
                    
                    uploaded += len(batch)
                    pbar.update(len(batch))
                    
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"\nUpload error: {e}")
                    continue
    
    print(f"\nUpload complete! Uploaded {uploaded:,} vectors")


# ============================================================
# VERIFICATION
# ============================================================

def verify_upload(chunks_collection, pinecone_index):
    """Verify upload completion"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # MongoDB stats
    total_chunks = chunks_collection.count_documents({})
    embedded_chunks = chunks_collection.count_documents({'embedded': True})
    synced_chunks = chunks_collection.count_documents({'synced_to_pinecone': True})
    
    print(f"\nMongoDB Atlas:")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Embedded: {embedded_chunks:,} ({embedded_chunks/total_chunks*100:.1f}%)")
    print(f"  Synced to Pinecone: {synced_chunks:,} ({synced_chunks/total_chunks*100:.1f}%)")
    
    # Pinecone stats
    stats = pinecone_index.describe_index_stats()
    vector_count = stats.get('total_vector_count', 0)
    
    print(f"\nPinecone ({INDEX_NAME}):")
    print(f"  Total vectors: {vector_count:,}")
    print(f"  Dimension: {EMBEDDING_DIMENSION}")
    print(f"  Metric: {METRIC}")
    
    if synced_chunks == embedded_chunks:
        print("\n✓ All embedded chunks are synced to Pinecone")
    else:
        print(f"\n⚠️  {embedded_chunks - synced_chunks:,} chunks not yet synced")


# ============================================================
# COST ESTIMATION
# ============================================================

def estimate_costs(chunks_collection):
    """Estimate Pinecone costs"""
    total_to_upload = chunks_collection.count_documents({
        'embedded': True,
        'synced_to_pinecone': False
    })
    
    if total_to_upload == 0:
        return
    
    print("\n" + "="*70)
    print("COST ESTIMATION")
    print("="*70)
    
    # Pinecone Serverless costs
    write_cost = (total_to_upload / 1_000_000) * 2.00  # $2 per 1M writes
    monthly_read_cost = 0.004  # Estimate for 10K queries/month
    
    print(f"\nPinecone Costs:")
    print(f"  One-time write: ${write_cost:.4f} ({total_to_upload:,} vectors)")
    print(f"  Monthly reads (10K queries): ~${monthly_read_cost:.2f}")
    print(f"\nTotal one-time: ${write_cost:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution"""
    print("="*70)
    print("MongoDB Atlas → Pinecone Upload")
    print("="*70)
    print()
    
    try:
        # Validate
        validate_config()
        
        # Setup
        chunks_collection = setup_mongodb()
        pinecone_index = setup_pinecone()
        
        # Cost estimation
        estimate_costs(chunks_collection)
        
        # Confirm
        total_to_upload = chunks_collection.count_documents({
            'embedded': True,
            'synced_to_pinecone': False
        })
        
        if total_to_upload > 0:
            print("\n" + "-"*70)
            response = input(f"\nUpload {total_to_upload:,} vectors to Pinecone? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled")
                return
            print()
        
        # Upload
        upload_to_pinecone(chunks_collection, pinecone_index)
        
        # Verify
        verify_upload(chunks_collection, pinecone_index)
        
        print("\n" + "="*70)
        print("UPLOAD COMPLETE")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Build RAG query API")
        print("  2. Test semantic search")
        print("  3. Integrate with LLM for answers")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        print("Progress is saved - you can resume later")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()