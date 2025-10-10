"""
MongoDB Atlas → Pinecone Embedding Upload
- Sadece embedding yükleme işlemi
- İki ayrı index: router (abstract) ve unified (full-text)
- Veri tekrarı yok
"""

import pymongo
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

# ============================================================
# KONFİGÜRASYON
# ============================================================

# MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medicrew")
CHUNKS_COLLECTION = "paper_chunks"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
ROUTER_INDEX_NAME = "medical-abstracts-router"
UNIFIED_INDEX_NAME = "medical-papers-unified"
EMBEDDING_DIMENSION = 768
METRIC = "cosine"

# İşlem
UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 500
RATE_LIMIT_DELAY = 0.5

# ============================================================
# BAĞLANTI KURMA
# ============================================================

def setup_mongodb():
    """MongoDB Atlas bağlantısı"""
    print("🔗 MongoDB Atlas'a bağlanılıyor...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        chunks_collection = db[CHUNKS_COLLECTION]
        client.server_info()
        print(f"✅ MongoDB bağlantısı başarılı: {CHUNKS_COLLECTION}")
        return chunks_collection
    except Exception as e:
        print(f"❌ MongoDB bağlantı hatası: {e}")
        exit(1)

def setup_pinecone_index(index_name: str, force_recreate: bool = False):
    """Pinecone index kurulumu"""
    print(f"\n🔄 Pinecone index hazırlanıyor: {index_name}")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            if force_recreate:
                print(f"⚠️ Mevcut index siliniyor: {index_name}")
                pc.delete_index(index_name)
                print("Index silindi. Bekleniyor...")
                time.sleep(10)
                
                print(f"Yeni index oluşturuluyor: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric=METRIC,
                    spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
                )
                print("✅ Yeni index oluşturuldu")
                time.sleep(2)
            else:
                print(f"✅ Mevcut index kullanılıyor: {index_name}")
        else:
            print(f"Yeni index oluşturuluyor: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            print(f"✅ Index oluşturuldu: {index_name}")
            time.sleep(2)
        
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        print(f"📊 Mevcut vektör sayısı ({index_name}): {vector_count:,}")
        
        return index
        
    except Exception as e:
        print(f"❌ Pinecone hatası ({index_name}): {e}")
        exit(1)

# ============================================================
# YÜKLEME FONKSİYONLARI
# ============================================================

def prepare_router_vector(chunk: Dict) -> Dict:
    """Router index için vektör hazırlama (sadece abstract'lar)"""
    vector_id = f"abstract_{chunk['pmid']}"
    
    metadata = {
        'pmid': chunk['pmid'],
        'title': chunk.get('title', '')[:500],
        'journal': chunk.get('journal', '')[:200],
        'year': chunk.get('year'),
        'chunk_type': 'abstract',
        'text_preview': chunk['text'][:200]
    }
    
    # None değerleri temizle
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return {
        'id': vector_id,
        'values': chunk['embedding'],
        'metadata': metadata
    }

def prepare_unified_vector(chunk: Dict) -> Dict:
    """Unified index için vektör hazırlama (sadece full-text chunk'lar)"""
    vector_id = f"{chunk['pmid']}_chunk_{chunk['chunk_index']}"
    
    metadata = {
        'pmid': chunk['pmid'],
        'chunk_index': chunk['chunk_index'],
        'chunk_type': 'full_text',
        'section': chunk.get('section', '')[:100],
        'title': chunk.get('title', '')[:500],
        'journal': chunk.get('journal', '')[:200],
        'year': chunk.get('year'),
        'text_preview': chunk['text'][:200],
        'token_count': chunk.get('token_count')
    }
    
    # None değerleri temizle
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return {
        'id': vector_id,
        'values': chunk['embedding'],
        'metadata': metadata
    }

def upload_to_router_index(chunks_collection, router_index):
    """Sadece ABSTRACT'ları router index'e yükle"""
    
    total_abstracts = chunks_collection.count_documents({
        'embedded': True,
        'is_abstract': True,
        'synced_to_router': False
    })
    
    if total_abstracts == 0:
        print("ℹ️ Router index için yeni abstract bulunamadı")
        return
    
    print(f"\n🚀 ROUTER INDEX: {total_abstracts:,} abstract yükleniyor...")
    
    uploaded = 0
    with tqdm(total=total_abstracts, desc="Abstract'lar", unit="vec") as pbar:
        while uploaded < total_abstracts:
            chunks = list(chunks_collection.find({
                'embedded': True,
                'is_abstract': True,
                'synced_to_router': False
            }, limit=FETCH_BATCH_SIZE))
            
            if not chunks:
                break
            
            for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
                batch = chunks[i:i + UPSERT_BATCH_SIZE]
                vectors = [prepare_router_vector(chunk) for chunk in batch]
                
                try:
                    router_index.upsert(vectors=vectors)
                    
                    # Sync flag'ini güncelle
                    chunk_ids = [chunk['_id'] for chunk in batch]
                    chunks_collection.update_many(
                        {'_id': {'$in': chunk_ids}},
                        {
                            '$set': {
                                'synced_to_router': True,
                                'router_synced_at': datetime.now(timezone.utc)
                            }
                        }
                    )
                    
                    uploaded += len(batch)
                    pbar.update(len(batch))
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"\n❌ Router yükleme hatası: {e}")
                    continue

def upload_to_unified_index(chunks_collection, unified_index):
    """Sadece FULL-TEXT chunk'larını unified index'e yükle"""
    
    total_chunks = chunks_collection.count_documents({
        'embedded': True,
        'is_abstract': False,
        'synced_to_unified': False
    })
    
    if total_chunks == 0:
        print("ℹ️ Unified index için yeni full-text chunk bulunamadı")
        return
    
    print(f"\n🎯 UNIFIED INDEX: {total_chunks:,} full-text chunk yükleniyor...")
    
    uploaded = 0
    with tqdm(total=total_chunks, desc="Full-Text", unit="vec") as pbar:
        while uploaded < total_chunks:
            chunks = list(chunks_collection.find({
                'embedded': True,
                'is_abstract': False,
                'synced_to_unified': False
            }, limit=FETCH_BATCH_SIZE))
            
            if not chunks:
                break
            
            for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
                batch = chunks[i:i + UPSERT_BATCH_SIZE]
                vectors = [prepare_unified_vector(chunk) for chunk in batch]
                
                try:
                    unified_index.upsert(vectors=vectors)
                    
                    # Sync flag'ini güncelle
                    chunk_ids = [chunk['_id'] for chunk in batch]
                    chunks_collection.update_many(
                        {'_id': {'$in': chunk_ids}},
                        {
                            '$set': {
                                'synced_to_unified': True,
                                'unified_synced_at': datetime.now(timezone.utc)
                            }
                        }
                    )
                    
                    uploaded += len(batch)
                    pbar.update(len(batch))
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"\n❌ Unified yükleme hatası: {e}")
                    continue

# ============================================================
# DOĞRULAMA
# ============================================================

def verify_upload(chunks_collection, router_index, unified_index):
    """Yükleme sonrası doğrulama"""
    print("\n" + "="*70)
    print("📊 YÜKLEME DOĞRULAMA")
    print("="*70)
    
    # MongoDB istatistikleri
    total_abstracts = chunks_collection.count_documents({'is_abstract': True})
    total_fulltext = chunks_collection.count_documents({'is_abstract': False})
    
    router_synced = chunks_collection.count_documents({'synced_to_router': True})
    unified_synced = chunks_collection.count_documents({'synced_to_unified': True})
    
    print(f"\n📈 MONGODB DURUMU:")
    print(f"  Toplam abstract: {total_abstracts:,}")
    print(f"  Toplam full-text: {total_fulltext:,}")
    print(f"  Router'a sync: {router_synced:,}/{total_abstracts:,}")
    print(f"  Unified'e sync: {unified_synced:,}/{total_fulltext:,}")
    
    # Pinecone istatistikleri
    router_stats = router_index.describe_index_stats()
    unified_stats = unified_index.describe_index_stats()
    
    print(f"\n🎯 PINECONE DURUMU:")
    print(f"  {ROUTER_INDEX_NAME}: {router_stats.get('total_vector_count', 0):,} vektör")
    print(f"  {UNIFIED_INDEX_NAME}: {unified_stats.get('total_vector_count', 0):,} vektör")
    
    if router_synced == total_abstracts and unified_synced == total_fulltext:
        print(f"\n✅ TÜM veriler başarıyla sync edildi!")
    else:
        print(f"\n⚠️  Sync tamamlanmadı:")
        if router_synced < total_abstracts:
            print(f"   - Router: {total_abstracts - router_synced:,} abstract eksik")
        if unified_synced < total_fulltext:
            print(f"   - Unified: {total_fulltext - unified_synced:,} chunk eksik")

def reset_sync_flags(chunks_collection):
    """Sync flag'lerini sıfırla (yeniden yükleme için)"""
    print("\n🔄 Sync flag'leri sıfırlanıyor...")
    
    chunks_collection.update_many(
        {},
        {
            '$set': {
                'synced_to_router': False,
                'synced_to_unified': False
            },
            '$unset': {
                'router_synced_at': '',
                'unified_synced_at': ''
            }
        }
    )
    
    print("✅ Sync flag'leri sıfırlandı")

# ============================================================
# ANA İŞLEM
# ============================================================

def main():
    """Ana yükleme işlemi"""
    print("="*70)
    print("MongoDB Atlas → Pinecone Embedding Yükleme")
    print("="*70)
    print("🎯 STRATEJİ:")
    print("   - ROUTER INDEX: Sadece abstract'lar")
    print("   - UNIFIED INDEX: Sadece full-text chunk'lar")
    print("   - VERİ TEKRARI: Yok")
    print()
    
    try:
        # Bağlantıları kur
        chunks_collection = setup_mongodb()
        
        # Kullanıcı seçimi
        print("\n" + "-"*70)
        print("Pinecone Seçenekleri:")
        print("1. Mevcut index'leri kullan (yeni vektörleri ekle)")
        print("2. TEMİZ BAŞLANGIÇ - Index'leri sil ve yeniden oluştur")
        
        choice = input("\nSeçim (1/2): ").strip()
        
        force_recreate = False
        if choice == "2":
            print("🚨 UYARI: Bu işlem tüm index'leri SİLECEK!")
            confirm = input("Onay için 'SIL' yazın: ")
            if confirm == "SIL":
                force_recreate = True
            else:
                print("❌ İşlem iptal edildi")
                return
        
        # Index'leri kur
        router_index = setup_pinecone_index(ROUTER_INDEX_NAME, force_recreate)
        unified_index = setup_pinecone_index(UNIFIED_INDEX_NAME, force_recreate)
        
        if force_recreate:
            reset_sync_flags(chunks_collection)
        
        # Yükleme işlemleri
        upload_to_router_index(chunks_collection, router_index)
        upload_to_unified_index(chunks_collection, unified_index)
        
        # Doğrulama
        verify_upload(chunks_collection, router_index, unified_index)
        
        print("\n" + "="*70)
        print("✅ EMBEDDING YÜKLEME TAMAMLANDI!")
        print("="*70)
        print("Sonraki adım: Retrieval pipeline'ını kurmak için hazır!")
        
    except KeyboardInterrupt:
        print("\n\n❌ İşlem kullanıcı tarafından iptal edildi")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()