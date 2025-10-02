"""
Hızlı Sistem Kontrolü
Sadece kritik şeyleri test eder (30 saniye)

python test_quick.py
"""

import pymongo
import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_test(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {name}: {status}")
    if details:
        print(f"     {details}")
    return passed

def main():
    print_header("HIZLI SISTEM KONTROLÜ")
    
    all_passed = True
    
    # 1. MongoDB Kontrolü
    print("\n1. MongoDB Bağlantısı")
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", 
                                     serverSelectionTimeoutMS=2000)
        client.server_info()
        all_passed &= print_test("MongoDB çalışıyor", True)
        client.close()
    except Exception as e:
        all_passed &= print_test("MongoDB çalışıyor", False, str(e))
        print("\nHATA: MongoDB başlatın (mongod)")
        return False
    
    # 2. Database & Collection Kontrolü
    print("\n2. Database Yapısı")
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["medicrew"]
        collection = db["pubmed_papers"]
        
        db_exists = "medicrew" in client.list_database_names()
        all_passed &= print_test("Database 'medicrew' var", db_exists)
        
        coll_exists = "pubmed_papers" in db.list_collection_names()
        all_passed &= print_test("Collection 'pubmed_papers' var", coll_exists)
        
        if not coll_exists:
            print("\nHATA: mongodb_initializer.py çalıştırın")
            return False
        
        client.close()
    except Exception as e:
        all_passed &= print_test("Database yapısı", False, str(e))
        return False
    
    # 3. Veri Kontrolü
    print("\n3. Veri Durumu")
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        collection = client["medicrew"]["pubmed_papers"]
        
        count = collection.count_documents({})
        has_data = count > 0
        all_passed &= print_test("Veri var", has_data, f"{count:,} paper")
        
        if count < 2000:
            print(f"     UYARI: Az veri var ({count}), 2000+ olmalı")
        
        # Domain dağılımı
        cardio = collection.count_documents({'domain': 'cardiology'})
        endo = collection.count_documents({'domain': 'endocrinology'})
        combined = collection.count_documents({'domain': 'combined'})
        
        all_passed &= print_test("Cardiology", cardio > 0, f"{cardio} paper")
        all_passed &= print_test("Endocrinology", endo > 0, f"{endo} paper")
        all_passed &= print_test("Combined", combined > 0, f"{combined} paper")
        
        # Full text
        full_text_count = collection.count_documents({'full_text': {'$ne': None}})
        percentage = (full_text_count / count * 100) if count > 0 else 0
        all_passed &= print_test("Full text", True, 
                                 f"{full_text_count} paper (%{percentage:.1f})")
        
        client.close()
    except Exception as e:
        all_passed &= print_test("Veri kontrolü", False, str(e))
        return False
    
    # 4. Index Kontrolü
    print("\n4. Index Durumu")
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        collection = client["medicrew"]["pubmed_papers"]
        
        indexes = list(collection.list_indexes())
        index_count = len(indexes)
        
        all_passed &= print_test("Index'ler oluşturulmuş", 
                                 index_count > 1, 
                                 f"{index_count} index")
        
        client.close()
    except Exception as e:
        all_passed &= print_test("Index kontrolü", False, str(e))
    
    # 5. Dosya Kontrolü
    print("\n5. Script Dosyaları")
    
    init_exists = os.path.exists('mongodb_initializer.py')
    all_passed &= print_test("mongodb_initializer.py", init_exists)
    
    downloader_exists = (os.path.exists('pubmed_mongodb_downloader.py') or 
                        os.path.exists('pubmed_downloader.py'))
    all_passed &= print_test("downloader script", downloader_exists)
    
    # Sonuç
    print_header("SONUÇ")
    
    if all_passed:
        print("\n✓ TÜM KONTROLLER BAŞARILI")
        print("\nSistem hazır durumda:")
        print("  1. MongoDB çalışıyor")
        print("  2. Veriler yüklü")
        print("  3. Index'ler mevcut")
        print("\nSONRAKI ADIM: Pinecone entegrasyonu")
    else:
        print("\n✗ BAZI KONTROLLER BAŞARISIZ")
        print("\nDüzeltilmesi gerekenler:")
        if not db_exists or not coll_exists:
            print("  - mongodb_initializer.py çalıştırın")
        if count < 100:
            print("  - pubmed_mongodb_downloader.py çalıştırın")
    
    print("\n" + "="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)