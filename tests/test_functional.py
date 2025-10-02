"""
Basit Functional Tests - Gerçek MongoDB ve dosyaları test eder
pytest test_functional.py -v
"""

import pytest
import pymongo
import os
from datetime import datetime

# Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
COLLECTION_NAME = "pubmed_papers"


class TestMongoDBConnection:
    """MongoDB bağlantı testleri"""
    
    def test_mongodb_is_running(self):
        """MongoDB server çalışıyor mu"""
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            client.server_info()
            client.close()
            assert True
        except Exception as e:
            pytest.fail(f"MongoDB çalışmıyor: {e}")
    
    def test_database_exists(self):
        """medicrew database var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        db_names = client.list_database_names()
        assert DB_NAME in db_names, f"Database '{DB_NAME}' bulunamadı"
        client.close()
    
    def test_collection_exists(self):
        """pubmed_papers collection var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collections = db.list_collection_names()
        assert COLLECTION_NAME in collections, f"Collection '{COLLECTION_NAME}' bulunamadı"
        client.close()


class TestDataQuality:
    """Veri kalitesi testleri"""
    
    def test_has_documents(self):
        """Collection'da document var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        count = collection.count_documents({})
        assert count > 0, "Collection boş"
        print(f"\n  Document count: {count}")
        client.close()
    
    def test_minimum_paper_count(self):
        """En az 2000 paper var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        count = collection.count_documents({})
        assert count >= 2000, f"Yetersiz veri: {count} (beklenen: >= 2000)"
        client.close()
    
    def test_required_fields_exist(self):
        """Gerekli field'lar var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # Rastgele bir document al
        sample = collection.find_one()
        assert sample is not None, "Document alınamadı"
        
        # Zorunlu field'lar
        required_fields = ['pmid', 'title', 'abstract', 'domain']
        for field in required_fields:
            assert field in sample, f"'{field}' field eksik"
        
        client.close()
    
    def test_domain_distribution(self):
        """Domain dağılımı dengeli mi"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        cardio_count = collection.count_documents({'domain': 'cardiology'})
        endo_count = collection.count_documents({'domain': 'endocrinology'})
        combined_count = collection.count_documents({'domain': 'combined'})
        
        print(f"\n  Cardiology: {cardio_count}")
        print(f"  Endocrinology: {endo_count}")
        print(f"  Combined: {combined_count}")
        
        # Her domain'de en az 300 paper olmalı
        assert cardio_count >= 300, f"Cardiology yetersiz: {cardio_count}"
        assert endo_count >= 300, f"Endocrinology yetersiz: {endo_count}"
        assert combined_count >= 100, f"Combined yetersiz: {combined_count}"
        
        client.close()
    
    def test_no_duplicate_pmids(self):
        """Duplicate PMID yok mu"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # PMID'leri say
        pipeline = [
            {'$group': {'_id': '$pmid', 'count': {'$sum': 1}}},
            {'$match': {'count': {'$gt': 1}}}
        ]
        
        duplicates = list(collection.aggregate(pipeline))
        assert len(duplicates) == 0, f"{len(duplicates)} duplicate PMID bulundu"
        
        client.close()
    
    def test_full_text_percentage(self):
        """Full text oranı makul mu"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        total = collection.count_documents({})
        with_full_text = collection.count_documents({'full_text': {'$ne': None}})
        
        percentage = (with_full_text / total) * 100
        print(f"\n  Full text oranı: {percentage:.1f}%")
        
        # En az %5 full text olmalı
        assert percentage >= 5, f"Full text oranı çok düşük: {percentage:.1f}%"
        
        client.close()


class TestIndexes:
    """Index testleri"""
    
    def test_indexes_exist(self):
        """Index'ler oluşturulmuş mu"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        indexes = list(collection.list_indexes())
        index_names = [idx['name'] for idx in indexes]
        
        print(f"\n  Bulunan index'ler: {index_names}")
        
        # En az _id index var olmalı
        assert '_id_' in index_names
        
        # PMID index olmalı
        pmid_indexes = [name for name in index_names if 'pmid' in name.lower()]
        assert len(pmid_indexes) > 0, "PMID index yok"
        
        client.close()
    
    def test_pmid_is_unique(self):
        """PMID unique index var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        indexes = list(collection.list_indexes())
        
        # PMID unique mı kontrol et
        pmid_unique = False
        for idx in indexes:
            if 'pmid' in idx.get('key', {}):
                if idx.get('unique', False):
                    pmid_unique = True
                    break
        
        assert pmid_unique, "PMID unique index değil"
        
        client.close()


class TestDataValidity:
    """Veri validasyon testleri"""
    
    def test_year_values_valid(self):
        """Year değerleri makul aralıkta mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # Year != null olan document'ler
        papers_with_year = list(collection.find(
            {'year': {'$ne': None}}, 
            {'year': 1}
        ).limit(100))
        
        for paper in papers_with_year:
            year = paper.get('year')
            if year:
                assert 2000 <= year <= 2026, f"Invalid year: {year}"
        
        client.close()
    
    def test_pmid_format_valid(self):
        """PMID formatı doğru mu"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # Rastgele 10 paper al
        samples = list(collection.find().limit(10))
        
        for paper in samples:
            pmid = paper.get('pmid')
            assert pmid, "PMID eksik"
            assert isinstance(pmid, str), f"PMID string değil: {type(pmid)}"
            assert len(pmid) >= 5, f"PMID çok kısa: {pmid}"
        
        client.close()
    
    def test_title_not_empty(self):
        """Title'lar boş değil mi"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        empty_titles = collection.count_documents({
            '$or': [
                {'title': None},
                {'title': ''},
                {'title': 'No Title'}
            ]
        })
        
        total = collection.count_documents({})
        percentage = (empty_titles / total) * 100
        
        # En fazla %1 boş title olabilir
        assert percentage < 1, f"Çok fazla boş title: {percentage:.1f}%"
        
        client.close()


class TestPineconeReadiness:
    """Pinecone'a yükleme için hazırlık testleri"""
    
    def test_synced_flag_exists(self):
        """synced_to_pinecone field var mı"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        sample = collection.find_one()
        assert 'synced_to_pinecone' in sample, "synced_to_pinecone field yok"
        
        client.close()
    
    def test_no_papers_synced_yet(self):
        """Henüz hiç paper sync edilmemiş mi"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        synced_count = collection.count_documents({'synced_to_pinecone': True})
        
        # İlk kez çalıştırıyoruz, hiç sync olmamalı
        # Eğer daha önce sync yaptıysanız, bu test fail edebilir
        print(f"\n  Synced paper count: {synced_count}")
        
        client.close()
    
    def test_text_content_for_embedding(self):
        """Embedding için text içeriği yeterli mi"""
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # Rastgele 10 paper al
        samples = list(collection.find().limit(10))
        
        for paper in samples:
            # Title + abstract birleşimi
            text_content = (paper.get('title', '') + ' ' + paper.get('abstract', '')).strip()
            
            # En az 50 karakter olmalı
            assert len(text_content) >= 50, f"PMID {paper.get('pmid')}: Text içeriği çok kısa"
        
        client.close()


class TestFileStructure:
    """Dosya yapısı testleri"""
    
    def test_main_scripts_exist(self):
        """Ana script dosyaları var mı"""
        required_files = [
            'mongodb_initializer.py',
        ]
        
        # pubmed_mongodb_downloader.py veya pubmed_downloader.py
        downloader_exists = (
            os.path.exists('pubmed_mongodb_downloader.py') or 
            os.path.exists('pubmed_downloader.py')
        )
        
        for file in required_files:
            assert os.path.exists(file), f"Dosya eksik: {file}"
        
        assert downloader_exists, "Downloader script bulunamadı"


# Test raporu
def pytest_sessionfinish(session, exitstatus):
    """Test bitince özet göster"""
    if exitstatus == 0:
        print("\n" + "="*70)
        print("BAŞARILI - Tüm testler geçti")
        print("Sistem Pinecone entegrasyonuna hazır")
        print("="*70)


if __name__ == "__main__":
    # Manuel çalıştırma
    pytest.main([__file__, "-v", "--tb=short", "-s"])