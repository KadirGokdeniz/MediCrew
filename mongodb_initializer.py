"""
MongoDB Database Initializer
MediCrew için MongoDB setup script
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

def test_mongodb_connection(uri):
    """MongoDB bağlantısını test et"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Ping to test connection
        client.admin.command('ping')
        print("✓ MongoDB connection successful")
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Çözümler:")
        print("   1. MongoDB çalışıyor mu kontrol et: mongod")
        print("   2. Connection string doğru mu: mongodb://localhost:27017/")
        print("   3. MongoDB kurulu mu: brew install mongodb-community (macOS)")
        sys.exit(1)


def create_database(client, db_name):
    """Database oluştur (ilk insert'te otomatik oluşur)"""
    db = client[db_name]
    print(f"✓ Database selected: {db_name}")
    return db


def create_collection(db, collection_name):
    """Collection oluştur ve yapılandır"""
    
    # Collection zaten varsa sil ve yeniden oluştur (temiz başlangıç)
    if collection_name in db.list_collection_names():
        print(f"⚠️  Collection '{collection_name}' zaten var")
        choice = input("   Silip yeniden oluşturmak ister misiniz? (y/n): ")
        if choice.lower() == 'y':
            db[collection_name].drop()
            print(f"   ✓ Eski collection silindi")
    
    # Collection oluştur (ilk insert'te otomatik oluşur)
    collection = db[collection_name]
    print(f"✓ Collection ready: {collection_name}")
    
    return collection


def create_indexes(collection):
    """Performance için index'ler oluştur"""
    
    print("\n📑 Creating indexes...")
    
    # 1. PMID (Unique) - primary key gibi
    collection.create_index(
        [("pmid", ASCENDING)],
        unique=True,
        name="idx_pmid_unique"
    )
    print("   ✓ Unique index: pmid")
    
    # 2. Domain (Filtering için)
    collection.create_index(
        [("domain", ASCENDING)],
        name="idx_domain"
    )
    print("   ✓ Index: domain")
    
    # 3. Year (Range queries için)
    collection.create_index(
        [("year", DESCENDING)],
        name="idx_year"
    )
    print("   ✓ Index: year")
    
    # 4. Pinecone sync durumu
    collection.create_index(
        [("synced_to_pinecone", ASCENDING)],
        name="idx_synced"
    )
    print("   ✓ Index: synced_to_pinecone")
    
    # 5. Text search (title + abstract)
    collection.create_index(
        [("title", TEXT), ("abstract", TEXT)],
        name="idx_text_search",
        default_language="english"
    )
    print("   ✓ Text index: title + abstract")
    
    # 6. Compound index (domain + year)
    collection.create_index(
        [("domain", ASCENDING), ("year", DESCENDING)],
        name="idx_domain_year"
    )
    print("   ✓ Compound index: domain + year")
    
    # 7. Downloaded_at (timestamp queries)
    collection.create_index(
        [("downloaded_at", DESCENDING)],
        name="idx_downloaded_at"
    )
    print("   ✓ Index: downloaded_at")
    
    print("\n✓ All indexes created")


def create_validation_schema(db, collection_name):
    """Document validation rules (optional ama önerilen)"""
    
    print("\n📋 Setting up validation schema...")
    
    # MongoDB'de schema validation (opsiyonel ama iyi practice)
    validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["pmid", "title"],  # Zorunlu field'lar
            "properties": {
                "pmid": {
                    "bsonType": "string",
                    "description": "PubMed ID - zorunlu ve unique"
                },
                "pmc_id": {
                    "bsonType": ["string", "null"],
                    "description": "PubMed Central ID - opsiyonel"
                },
                "title": {
                    "bsonType": "string",
                    "description": "Paper title - zorunlu"
                },
                "abstract": {
                    "bsonType": ["string", "null"],
                    "description": "Paper abstract"
                },
                "full_text": {
                    "bsonType": ["string", "null"],
                    "description": "Full article text - opsiyonel"
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
                    "description": "Authors - string veya array"
                },
                "domain": {
                    "enum": ["cardiology", "endocrinology", "combined", None],
                    "description": "Medical domain"
                },
                "synced_to_pinecone": {
                    "bsonType": "bool",
                    "description": "Pinecone'a yüklendi mi?"
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
    
    # Validation'ı uygula
    try:
        db.command({
            "collMod": collection_name,
            "validator": validator,
            "validationLevel": "moderate"  # moderate = mevcut doc'lar hariç
        })
        print("✓ Validation schema applied")
    except Exception as e:
        print(f"⚠️  Validation schema uygulanamadı (normal - ilk kez): {e}")


def insert_sample_data(collection):
    """Test için sample data ekle"""
    
    print("\n📝 Inserting sample documents...")
    
    sample_papers = [
        {
            "pmid": "99999991",
            "title": "Sample Paper: Heart Failure Treatment Guidelines",
            "abstract": "This is a sample abstract for testing purposes...",
            "journal": "Test Journal of Medicine",
            "year": 2024,
            "authors": "Test Author A, Test Author B",
            "domain": "cardiology",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/99999991/",
            "synced_to_pinecone": False,
            "downloaded_at": datetime.utcnow(),
            "metadata": {
                "query": "test query",
                "has_full_text": False,
                "is_sample": True
            }
        },
        {
            "pmid": "99999992",
            "title": "Sample Paper: Diabetes Management in 2024",
            "abstract": "This is another sample abstract...",
            "journal": "Test Endocrinology Review",
            "year": 2024,
            "authors": ["Smith J", "Doe A"],  # Array format
            "domain": "endocrinology",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/99999992/",
            "synced_to_pinecone": False,
            "downloaded_at": datetime.utcnow(),
            "metadata": {
                "query": "diabetes test",
                "has_full_text": False,
                "impact_factor": 8.5,
                "is_sample": True
            }
        },
        {
            "pmid": "99999993",
            "title": "Sample Paper: Diabetic Cardiomyopathy",
            "abstract": "Combined domain sample...",
            "full_text": "This paper has full text content for testing...",
            "journal": "Test Combined Medicine",
            "year": 2023,
            "authors": "Johnson M, Lee K, Park S",
            "domain": "combined",
            "pmc_id": "PMC9999999",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/99999993/",
            "pmc_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9999999/",
            "synced_to_pinecone": False,
            "downloaded_at": datetime.utcnow(),
            "metadata": {
                "query": "combined test",
                "has_full_text": True,
                "citations_count": 42,
                "is_sample": True
            }
        }
    ]
    
    try:
        result = collection.insert_many(sample_papers)
        print(f"✓ Inserted {len(result.inserted_ids)} sample documents")
        return True
    except Exception as e:
        print(f"⚠️  Sample data insertion failed: {e}")
        return False


def print_database_stats(db, collection):
    """Database istatistiklerini göster"""
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
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
        print(f"   - {idx['name']}: {idx['key']}")
    
    # Sample documents
    if doc_count > 0:
        print(f"\n📄 Sample document:")
        sample = collection.find_one()
        if sample:
            for key, value in list(sample.items())[:8]:  # İlk 8 field
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"   {key}: {value}")


def main():
    """Ana fonksiyon - tüm setup işlemleri"""
    
    print("="*60)
    print("🏥 MongoDB Database Initializer - MediCrew")
    print("="*60)
    print()
    
    # 1. MongoDB bağlantısı
    client = test_mongodb_connection(MONGO_URI)
    
    # 2. Database oluştur
    db = create_database(client, DB_NAME)
    
    # 3. Collection oluştur
    collection = create_collection(db, COLLECTION_NAME)
    
    # 4. Index'leri oluştur
    create_indexes(collection)
    
    # 5. Validation schema (opsiyonel)
    create_validation_schema(db, COLLECTION_NAME)
    
    # 6. Sample data ekle (test için)
    print()
    insert_sample = input("Sample test data eklemek ister misiniz? (y/n): ")
    if insert_sample.lower() == 'y':
        insert_sample_data(collection)
    
    # 7. İstatistikleri göster
    print_database_stats(db, collection)
    
    print("\n" + "="*60)
    print("✅ MongoDB Database Initialization Complete!")
    print("="*60)
    print(f"\n📍 Connection String: {MONGO_URI}")
    print(f"📍 Database: {DB_NAME}")
    print(f"📍 Collection: {COLLECTION_NAME}")
    print("\n💡 Next Steps:")
    print("   1. MongoDB Compass'ta görüntüleyin: mongodb://localhost:27017")
    print("   2. PubMed downloader script'ini çalıştırın")
    print("   3. Pinecone'a embedding'leri yükleyin")
    print("\n🚀 Hazırsınız!")
    
    client.close()


if __name__ == "__main__":
    main()