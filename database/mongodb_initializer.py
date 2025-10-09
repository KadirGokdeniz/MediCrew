"""
MongoDB Atlas Database Initializer
Prepares database structure before data ingestion
to fresh start 
python database/mongodb_initializer.py --reset --force
"""

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
import sys
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medicrew")
COLLECTIONS = {
    "pubmed_papers": "Papers from PubMed",
    "paper_chunks": "Chunked papers for embeddings"
}

if not MONGO_URI:
    print("ERROR: MONGO_URI not found in .env file!")
    sys.exit(1)

# ============================================================
# CONNECTION
# ============================================================

def connect_to_mongodb(uri):
    """Connect to MongoDB Atlas"""
    print("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        client.admin.command('ping')
        print("✓ Connected successfully\n")
        return client
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check MONGO_URI in .env")
        print("  2. Verify IP whitelist in Atlas (0.0.0.0/0)")
        print("  3. Verify user credentials")
        sys.exit(1)


# ============================================================
# CLEANUP
# ============================================================

def cleanup_collections(db, force=False):
    """Drop existing collections if present"""
    
    existing = db.list_collection_names()
    to_drop = [c for c in COLLECTIONS.keys() if c in existing]
    
    if not to_drop:
        print("No existing collections found\n")
        return
    
    print(f"Found existing collections: {', '.join(to_drop)}")
    
    if not force:
        print("\nWARNING: This will delete all data in these collections!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            sys.exit(0)
    
    print()
    for collection_name in to_drop:
        db[collection_name].drop()
        print(f"✓ Dropped: {collection_name}")
    
    print()


# ============================================================
# INDEX MANAGEMENT
# ============================================================

def clean_conflicting_indexes(collection):
    """Remove old indexes that might conflict"""
    try:
        existing_indexes = list(collection.list_indexes())
        
        # Remove old-style indexes
        for idx in existing_indexes:
            if idx['name'] in ['idx_pmid_unique', 'pmid_1']:
                try:
                    collection.drop_index(idx['name'])
                    print(f"  Removed old index: {idx['name']}")
                except:
                    pass
    except Exception as e:
        pass  # No existing indexes

def create_pubmed_papers_indexes(collection):
    """Create indexes for pubmed_papers collection"""
    print("Creating indexes for pubmed_papers...")
    
    clean_conflicting_indexes(collection)
    
    indexes_created = 0
    
    try:
        collection.create_index([("pmid", ASCENDING)], unique=True, name="idx_pmid")
        print("  ✓ pmid (unique)")
        indexes_created += 1
    except pymongo.errors.OperationFailure:
        print("  - pmid (already exists)")
    
    try:
        collection.create_index([("domain", ASCENDING)], name="idx_domain")
        print("  ✓ domain")
        indexes_created += 1
    except:
        print("  - domain (already exists)")
    
    try:
        collection.create_index([("year", DESCENDING)], name="idx_year")
        print("  ✓ year")
        indexes_created += 1
    except:
        print("  - year (already exists)")
    
    try:
        collection.create_index([("synced_to_pinecone", ASCENDING)], name="idx_synced")
        print("  ✓ synced_to_pinecone")
        indexes_created += 1
    except:
        print("  - synced_to_pinecone (already exists)")
    
    try:
        collection.create_index(
            [("title", TEXT), ("abstract", TEXT)], 
            name="idx_text_search",
            default_language="english"
        )
        print("  ✓ text search (title + abstract)")
        indexes_created += 1
    except:
        print("  - text search (already exists)")
    
    try:
        collection.create_index(
            [("domain", ASCENDING), ("year", DESCENDING)], 
            name="idx_domain_year"
        )
        print("  ✓ domain + year (compound)")
        indexes_created += 1
    except:
        print("  - domain + year (already exists)")
    
    print(f"Total: {indexes_created} new indexes created\n")


def create_paper_chunks_indexes(collection):
    """Create indexes for paper_chunks collection"""
    print("Creating indexes for paper_chunks...")
    
    clean_conflicting_indexes(collection)
    
    indexes_created = 0
    
    try:
        collection.create_index([("pmid", ASCENDING)], name="idx_pmid")
        print("  ✓ pmid")
        indexes_created += 1
    except:
        print("  - pmid (already exists)")
    
    try:
        collection.create_index(
            [("pmid", ASCENDING), ("chunk_index", ASCENDING)], 
            unique=True, 
            name="idx_pmid_chunk"
        )
        print("  ✓ pmid + chunk_index (unique)")
        indexes_created += 1
    except:
        print("  - pmid + chunk_index (already exists)")
    
    try:
        collection.create_index([("chunk_type", ASCENDING)], name="idx_chunk_type")
        print("  ✓ chunk_type")
        indexes_created += 1
    except:
        print("  - chunk_type (already exists)")
    
    try:
        collection.create_index([("domain", ASCENDING)], name="idx_domain")
        print("  ✓ domain")
        indexes_created += 1
    except:
        print("  - domain (already exists)")
    
    try:
        collection.create_index([("embedded", ASCENDING)], name="idx_embedded")
        print("  ✓ embedded")
        indexes_created += 1
    except:
        print("  - embedded (already exists)")
    
    try:
        collection.create_index([("synced_to_pinecone", ASCENDING)], name="idx_synced")
        print("  ✓ synced_to_pinecone")
        indexes_created += 1
    except:
        print("  - synced_to_pinecone (already exists)")
    
    print(f"Total: {indexes_created} new indexes created\n")

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_collections(db):
    """Initialize all collections with indexes"""
    
    print("="*60)
    print("INITIALIZING COLLECTIONS")
    print("="*60)
    print()
    
    # pubmed_papers
    papers_collection = db["pubmed_papers"]
    create_pubmed_papers_indexes(papers_collection)
    
    # paper_chunks
    chunks_collection = db["paper_chunks"]
    create_paper_chunks_indexes(chunks_collection)


# ============================================================
# STATISTICS
# ============================================================

def show_status(db):
    """Show current database status"""
    
    print("="*60)
    print("DATABASE STATUS")
    print("="*60)
    print()
    
    for collection_name, description in COLLECTIONS.items():
        if collection_name in db.list_collection_names():
            collection = db[collection_name]
            doc_count = collection.count_documents({})
            index_count = len(list(collection.list_indexes()))
            
            print(f"{collection_name}:")
            print(f"  Description: {description}")
            print(f"  Documents: {doc_count:,}")
            print(f"  Indexes: {index_count}")
            print(f"  Status: {'✓ Ready' if index_count > 1 else '⚠ Not initialized'}")
            print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Initialize MongoDB Atlas structure for MediCrew'
    )
    parser.add_argument(
        '--reset', 
        action='store_true', 
        help='Drop existing collections before initialization'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Skip confirmation prompts (for Docker/CI)'
    )
    
    args = parser.parse_args()
    
    print()
    print("="*60)
    print("MongoDB Atlas Initializer - MediCrew")
    print("="*60)
    print()
    print("Purpose: Prepare database structure before data ingestion")
    print()
    
    # Connect
    client = connect_to_mongodb(MONGO_URI)
    db = client[DB_NAME]
    
    # Cleanup if requested
    if args.reset:
        cleanup_collections(db, force=args.force)
    
    # Initialize
    initialize_collections(db)
    
    # Show status
    show_status(db)
    
    print("="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print()
    print("Database structure is ready!")
    
    client.close()

if __name__ == "__main__":
    main()