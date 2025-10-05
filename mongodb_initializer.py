"""
MongoDB Database Initializer
Setup script for MediCrew database
"""

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from datetime import datetime
import sys

# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
COLLECTION_NAME = "pubmed_papers"


# ============================================================
# MongoDB SETUP & INITIALIZATION
# ============================================================

def check_mongodb_connection(uri):
    """Test MongoDB connection"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB connection successful")
        return client
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if MongoDB is running: mongod")
        print("  2. Verify connection string: mongodb://localhost:27017/")
        print("  3. Install MongoDB: https://www.mongodb.com/docs/manual/installation/")
        sys.exit(1)


def create_database(client, db_name):
    """Create database (automatically created on first insert)"""
    db = client[db_name]
    print(f"Database selected: {db_name}")
    return db


def create_collection(db, collection_name):
    """Create and configure collection"""
    
    # Drop existing collection if present
    if collection_name in db.list_collection_names():
        print(f"Warning: Collection '{collection_name}' already exists")
        db[collection_name].drop()
        print("Existing collection dropped")
    
    # Collection will be created on first insert
    collection = db[collection_name]
    print(f"Collection ready: {collection_name}")
    
    return collection


def create_indexes(collection):
    """Create indexes for performance optimization"""
    
    print("\nCreating indexes...")
    
    # 1. PMID (Unique) - acts as primary key
    collection.create_index(
        [("pmid", ASCENDING)],
        unique=True
    )
    print("  Created unique index: pmid")
    
    # 2. Domain (for filtering)
    collection.create_index(
        [("domain", ASCENDING)],
        name="idx_domain"
    )
    print("  Created index: domain")
    
    # 3. Year (for range queries)
    collection.create_index(
        [("year", DESCENDING)],
        name="idx_year"
    )
    print("  Created index: year")
    
    # 4. Pinecone sync status
    collection.create_index(
        [("synced_to_pinecone", ASCENDING)],
        name="idx_synced"
    )
    print("  Created index: synced_to_pinecone")
    
    # 5. Text search (title + abstract)
    collection.create_index(
        [("title", TEXT), ("abstract", TEXT)],
        name="idx_text_search",
        default_language="english"
    )
    print("  Created text index: title + abstract")
    
    # 6. Compound index (domain + year)
    collection.create_index(
        [("domain", ASCENDING), ("year", DESCENDING)],
        name="idx_domain_year"
    )
    print("  Created compound index: domain + year")
    
    # 7. Downloaded_at (for timestamp queries)
    collection.create_index(
        [("downloaded_at", DESCENDING)],
        name="idx_downloaded_at"
    )
    print("  Created index: downloaded_at")
    
    print("\nAll indexes created successfully")


def create_validation_schema(db, collection_name):
    """Setup document validation rules"""
    
    print("\nSetting up validation schema...")
    
    validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["pmid", "title"],
            "properties": {
                "pmid": {
                    "bsonType": "string",
                    "description": "PubMed ID - required and unique"
                },
                "pmc_id": {
                    "bsonType": ["string", "null"],
                    "description": "PubMed Central ID - optional"
                },
                "title": {
                    "bsonType": "string",
                    "description": "Paper title - required"
                },
                "abstract": {
                    "bsonType": ["string", "null"],
                    "description": "Paper abstract"
                },
                "full_text": {
                    "bsonType": ["string", "null"],
                    "description": "Full article text - optional"
                },
                "year": {
                    "bsonType": ["int", "null"],
                    "minimum": 1900,
                    "maximum": 2100,
                    "description": "Publication year"
                },
                "journal": {
                    "bsonType": ["string", "null"],
                    "description": "Journal name"
                },
                "authors": {
                    "bsonType": ["string", "array", "null"],
                    "description": "Authors - string or array"
                },
                "domain": {
                    "enum": ["cardiology", "endocrinology", "combined", None],
                    "description": "Medical domain"
                },
                "synced_to_pinecone": {
                    "bsonType": "bool",
                    "description": "Synced to Pinecone?"
                },
                "downloaded_at": {
                    "bsonType": "date",
                    "description": "Download timestamp"
                },
                "metadata": {
                    "bsonType": ["object", "null"],
                    "description": "Extra flexible data"
                }
            }
        }
    }
    
    try:
        db.command({
            "collMod": collection_name,
            "validator": validator,
            "validationLevel": "moderate"
        })
        print("Validation schema applied successfully")
    except Exception as e:
        print(f"Warning: Validation schema could not be applied: {e}")


def print_database_stats(db, collection):
    """Display database statistics"""
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    # Database size
    stats = db.command("dbStats")
    print(f"Database: {db.name}")
    print(f"Collections: {len(db.list_collection_names())}")
    print(f"Database size: {stats['dataSize'] / 1024:.2f} KB")
    
    # Collection stats
    coll_stats = db.command("collStats", collection.name)
    doc_count = coll_stats['count']
    print(f"\nCollection: {collection.name}")
    print(f"Documents: {doc_count}")
    print(f"Avg document size: {coll_stats.get('avgObjSize', 0) / 1024:.2f} KB")
    print(f"Total size: {coll_stats.get('size', 0) / 1024:.2f} KB")
    
    # Indexes
    indexes = collection.list_indexes()
    print(f"\nIndexes:")
    for idx in indexes:
        print(f"  - {idx['name']}: {idx['key']}")
    
    # Sample document (if exists)
    if doc_count > 0:
        print(f"\nSample document:")
        sample = collection.find_one()
        if sample:
            for key, value in list(sample.items())[:8]:
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"  {key}: {value}")


def main():
    """Main initialization function"""
    
    print("="*60)
    print("MongoDB Database Initializer - MediCrew")
    print("="*60)
    print()
    
    # 1. Test MongoDB connection
    client = check_mongodb_connection(MONGO_URI)
    
    # 2. Create database
    db = create_database(client, DB_NAME)
    
    # 3. Create collection
    collection = create_collection(db, COLLECTION_NAME)
    
    # 4. Create indexes
    create_indexes(collection)
    
    # 5. Setup validation schema
    create_validation_schema(db, COLLECTION_NAME)
    
    # 6. Display statistics
    print_database_stats(db, collection)
    
    print("\n" + "="*60)
    print("MongoDB Database Initialization Complete")
    print("="*60)
    print(f"\nConnection String: {MONGO_URI}")
    print(f"Database: {DB_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print("\nNext Steps:")
    print("  1. View in MongoDB Compass: mongodb://localhost:27017")
    print("  2. Run PubMed downloader script")
    print("  3. Upload embeddings to Pinecone")
    print("\nReady to proceed!")
    
    client.close()


if __name__ == "__main__":
    main()