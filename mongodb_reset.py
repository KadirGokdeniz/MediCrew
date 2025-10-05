"""
MongoDB Reset & Clean Setup
Temizler ve sıfırdan başlatır
"""

import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medicrew")

def reset_database():
    """MongoDB'yi tamamen temizle ve hazırla"""
    
    print("="*70)
    print("MongoDB Reset & Clean Setup")
    print("="*70)
    print()
    
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB'ye bağlandı")
    except Exception as e:
        print(f"❌ MongoDB'ye bağlanılamadı: {e}")
        return
    
    db = client[DB_NAME]
    
    # Mevcut collection'ları listele
    collections = db.list_collection_names()
    print(f"\nMevcut collection'lar: {collections}")
    
    # pubmed_papers collection'ını tamamen sil
    if "pubmed_papers" in collections:
        print("\n⚠️  pubmed_papers collection'ı SİLİNİYOR...")
        response = input("Emin misiniz? (evet/hayır): ")
        
        if response.lower() == 'evet':
            db.pubmed_papers.drop()
            print("✅ pubmed_papers silindi")
        else:
            print("❌ İşlem iptal edildi")
            return
    
    # paper_chunks collection'ını da sil (varsa)
    if "paper_chunks" in collections:
        print("\n⚠️  paper_chunks collection'ı da SİLİNİYOR...")
        db.paper_chunks.drop()
        print("✅ paper_chunks silindi")
    
    print("\n" + "="*70)
    print("✅ MongoDB tamamen temizlendi!")
    print("="*70)
    print("\nŞimdi yapılacaklar:")
    print("  1. python pubmed_downloader.py çalıştır")
    print("  2. python hybrid_chunking.py çalıştır")
    print("  3. python pinecone_integration.py çalıştır")
    print()
    
    client.close()

if __name__ == "__main__":
    reset_database()