"""
Download Cardiology Papers Script
Contains execution logic and loads queries from JSON file
"""

import os
import json
from dotenv import load_dotenv
import pymongo
from pubmed_downloader import download_to_mongodb

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# NCBI Entrez Configuration
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medicrew")
COLLECTION_NAME = "pubmed_papers"

# Path to queries JSON file
QUERIES_JSON_PATH = os.path.join(os.path.dirname(__file__), "cardio_queries.json")

# ============================================================
# QUERY LOADING FUNCTIONS
# ============================================================

def load_queries_from_json():
    """
    Load cardiology queries from JSON file
    Returns a dictionary of category: [queries]
    """
    try:
        with open(QUERIES_JSON_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        # Verify successful loading
        print(f"📋 Loaded queries for {len(queries_data)} categories")
        for category, queries in queries_data.items():
            print(f"  - {category}: {len(queries)} queries")
        
        return queries_data
    except FileNotFoundError:
        print(f"❌ Error: Query file not found at {QUERIES_JSON_PATH}")
        print("💡 Create the file or update the path in the configuration")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in query file: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading queries: {e}")
        exit(1)

def get_all_queries(queries_data):
    """Return all cardiology queries as a flat list of query strings"""
    all_queries = []
    for category, queries in queries_data.items():
        for query_obj in queries:
            all_queries.append(query_obj["query"])
    return all_queries

def get_queries_by_category(queries_data, category):
    """Return queries for a specific category as list of query strings"""
    if category not in queries_data:
        return []
    
    return [query_obj["query"] for query_obj in queries_data[category]]

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Validate required environment variables"""
    if not ENTREZ_EMAIL:
        print("❌ ERROR: ENTREZ_EMAIL not set in .env file!")
        print("\n💡 Create a .env file with:")
        print("   ENTREZ_EMAIL=your_email@example.com")
        exit(1)
    
    if not ENTREZ_API_KEY:
        print("⚠️ WARNING: ENTREZ_API_KEY not set")
        print("   Downloads will be slower (3 req/sec vs 10 req/sec)")
        print("   Get your key: https://www.ncbi.nlm.nih.gov/account/settings/")
        print()
    else:
        print("✅ API Key detected - faster downloads enabled")

# ============================================================
# MONGODB CONNECTION
# ============================================================

def connect_mongodb():
    """Connect to MongoDB"""
    print("🔌 Connecting to MongoDB...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Test connection
        client.server_info()
        print(f"✅ Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        
        # Show existing document count
        existing_count = collection.count_documents({})
        print(f"📊 Existing documents: {existing_count}")
        
        return collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Solutions:")
        print("   1. Is MongoDB running? Terminal: mongod")
        print("   2. Check MONGO_URI in .env file")
        exit(1)

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function - all download operations"""
    
    print("="*70)
    print("🥼 MediCrew PubMed → MongoDB Downloader (Cardiology Focus)")
    print("="*70)
    print()
    
    # Validate configuration
    validate_config()
    print()
    
    # Load queries from JSON
    cardio_queries = load_queries_from_json()
    print()
    
    # MongoDB connection
    collection = connect_mongodb()
    print()
    
    total_saved = 0
    
    # ================================================================
    # CARDIOLOGY PAPERS DOWNLOAD - EITHER BY CATEGORY OR ALL
    # ================================================================
    
    # Option 1: Download by category
    categories = list(cardio_queries.keys())
    
    for category in categories:
        print(f"\n{'='*70}")
        print(f"📋 CATEGORY: {category.upper()}")
        print(f"{'='*70}")
        
        queries = get_queries_by_category(cardio_queries, category)
        
        for query in queries:
            count = download_to_mongodb(
                collection, 
                query, 
                'cardiology', 
                max_results=40,
                email=ENTREZ_EMAIL,
                api_key=ENTREZ_API_KEY
            )
            total_saved += count
            print()
    
    # Option 2: Download all queries at once (uncomment to use)
    """
    print(f"\n{'='*70}")
    print(f"📋 ALL CARDIOLOGY QUERIES")
    print(f"{'='*70}")
    
    for query in get_all_queries(cardio_queries):
        count = download_to_mongodb(
            collection, 
            query, 
            'cardiology', 
            max_results=40,
            email=ENTREZ_EMAIL,
            api_key=ENTREZ_API_KEY
        )
        total_saved += count
        print()
    """
    
    # ================================================================
    # FINAL STATISTICS
    # ================================================================
    print("\n" + "="*70)
    print("📊 FINAL STATISTICS")
    print("="*70)
    
    # Total counts
    total_count = collection.count_documents({})
    full_text_count = collection.count_documents({'full_text': {'$ne': None}})
    
    if total_count > 0:
        print(f"\nTotal papers in MongoDB: {total_count}")
        print(f"Papers with full text: {full_text_count} ({full_text_count/total_count*100:.1f}%)")
        print(f"Papers with abstract only: {total_count - full_text_count}")
        
        # Only cardiology domain now
        cardio_count = collection.count_documents({'domain': 'cardiology'})
        print(f"\nCardiology papers: {cardio_count}")
        
        # Year distribution
        print("\nYear distribution:")
        pipeline = [
            {'$match': {'year': {'$ne': None}}},
            {'$group': {'_id': '$year', 'count': {'$sum': 1}}},
            {'$sort': {'_id': -1}},
            {'$limit': 5}
        ]
        year_stats = list(collection.aggregate(pipeline))
        for stat in year_stats:
            print(f"  - {stat['_id']}: {stat['count']} papers")
    else:
        print("\nNo papers in the database yet.")
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"📁 Database: {DB_NAME}")
    print(f"📁 Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()