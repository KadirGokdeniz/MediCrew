"""
PubMed to MongoDB Downloader
MediCrew için - MongoDB'ye direkt veri indirme
"""

from Bio import Entrez
import time
import pymongo
from tqdm import tqdm
from datetime import datetime

# ============================================================
# ⚠️ CONFIGURASYON - BURALAYI DEĞİŞTİRİN!
# ============================================================

Entrez.email = "kadirqokdeniz@hotmail.com"  # ← KENDİ EMAİLİNİZİ GİRİN!
# Entrez.api_key = "YOUR_API_KEY"  # Opsiyonel - daha hızlı indirme için

# MongoDB bağlantısı
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
COLLECTION_NAME = "pubmed_papers"

# ============================================================
# MONGODB CONNECTION
# ============================================================

def connect_mongodb():
    """MongoDB'ye bağlan"""
    print("🔌 Connecting to MongoDB...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Test connection
        client.server_info()
        print(f"✓ Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        
        # Mevcut document sayısı
        existing_count = collection.count_documents({})
        print(f"✓ Existing documents: {existing_count}")
        
        return collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Çözümler:")
        print("   1. MongoDB çalışıyor mu? Terminal: mongod")
        print("   2. MongoDB Initializer çalıştırdınız mı?")
        exit(1)


# ============================================================
# PUBMED API FUNCTIONS
# ============================================================

def search_pubmed(query, max_results=500):
    """PubMed'de arama yap, PMID listesi döndür"""
    print(f"🔍 Searching: '{query}' (max {max_results} results)...")
    
    try:
        handle = Entrez.esearch(
            db='pubmed',
            term=query,
            retmax=max_results,
            sort='relevance',
            retmode='xml'
        )
        results = Entrez.read(handle)
        handle.close()
        
        pmids = results['IdList']
        print(f"✓ Found {len(pmids)} articles")
        return pmids
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def get_pmc_id(pmid):
    """PMID'den PMC ID bul (full text için)"""
    try:
        handle = Entrez.elink(
            dbfrom='pubmed',
            db='pmc',
            id=pmid,
            linkname='pubmed_pmc'
        )
        record = Entrez.read(handle)
        handle.close()
        
        if record[0]['LinkSetDb']:
            pmc_ids = [link['Id'] for link in record[0]['LinkSetDb'][0]['Link']]
            return pmc_ids[0] if pmc_ids else None
        return None
    except:
        return None


def fetch_full_text_from_pmc(pmc_id):
    """PMC'den full text çek (Open Access için)"""
    try:
        handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
        )
        xml_content = handle.read()
        handle.close()
        
        # XML'i text'e çevir (basitleştirilmiş)
        if xml_content and len(xml_content) > 1000:
            text = xml_content.decode('utf-8', errors='ignore')
            # İlk 50k karakter (MongoDB document limit için)
            return text[:50000] if len(text) > 50000 else text
        return None
    except Exception as e:
        return None


def fetch_paper_details(pmid):
    """Tek paper'ın tüm detaylarını çek"""
    try:
        # PubMed'den basic bilgileri al
        handle = Entrez.efetch(
            db='pubmed',
            id=pmid,
            retmode='xml'
        )
        records = Entrez.read(handle)
        handle.close()
        
        if not records.get('PubmedArticle'):
            return None
        
        record = records['PubmedArticle'][0]
        medline = record['MedlineCitation']
        article = medline['Article']
        
        # PMID
        pmid_str = str(medline['PMID'])
        
        # Title
        title = str(article.get('ArticleTitle', 'No Title'))
        
        # Abstract
        abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_parts, list):
            abstract = ' '.join([str(part) for part in abstract_parts])
        else:
            abstract = str(abstract_parts) if abstract_parts else ''
        
        # Journal
        journal = article.get('Journal', {}).get('Title', 'Unknown')
        
        # Year
        pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        year_str = pub_date.get('Year', None)
        year = int(year_str) if year_str and year_str.isdigit() else None
        
        # Authors
        author_list = article.get('AuthorList', [])
        authors = []
        for author in author_list[:5]:  # İlk 5 yazar
            last = author.get('LastName', '')
            init = author.get('Initials', '')
            if last:
                authors.append(f"{last} {init}".strip())
        authors_str = ', '.join(authors) if authors else 'Unknown'
        
        # URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
        
        # Full text dene (PMC'den)
        full_text = None
        pmc_id = get_pmc_id(pmid_str)
        pmc_url = None
        
        if pmc_id:
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            full_text = fetch_full_text_from_pmc(pmc_id)
            # Rate limit için ekstra bekleme
            time.sleep(0.2)
        
        # Paper objesi oluştur
        paper = {
            'pmid': pmid_str,
            'pmc_id': pmc_id,
            'title': title,
            'abstract': abstract,
            'full_text': full_text,
            'journal': journal,
            'year': year,
            'authors': authors_str,
            'pubmed_url': pubmed_url,
            'pmc_url': pmc_url,
            'downloaded_at': datetime.utcnow(),
            'synced_to_pinecone': False,  # MongoDB initializer ile uyumlu
            'metadata': {
                'has_full_text': full_text is not None,
                'pmc_available': pmc_id is not None
            }
        }
        
        return paper
        
    except Exception as e:
        print(f"⚠️  Error fetching PMID {pmid}: {e}")
        return None


# ============================================================
# MONGODB OPERATIONS
# ============================================================

def download_to_mongodb(collection, query, domain, max_results=300):
    """PubMed'den MongoDB'ye direkt kaydet"""
    
    # Arama yap
    pmids = search_pubmed(query, max_results)
    
    if not pmids:
        print("⚠️  No results found")
        return 0
    
    saved_count = 0
    skipped_count = 0
    error_count = 0
    full_text_count = 0
    
    print(f"📥 Downloading {len(pmids)} papers...")
    
    # Progress bar ile işle
    for pmid in tqdm(pmids, desc="Processing"):
        try:
            # Duplicate kontrolü
            if collection.find_one({'pmid': str(pmid)}):
                skipped_count += 1
                continue
            
            # Paper detaylarını çek
            paper = fetch_paper_details(pmid)
            
            if not paper:
                error_count += 1
                continue
            
            # Domain ve query bilgilerini ekle
            paper['domain'] = domain
            paper['metadata']['query'] = query
            
            # MongoDB'ye kaydet
            collection.insert_one(paper)
            saved_count += 1
            
            # Full text varsa say
            if paper.get('full_text'):
                full_text_count += 1
            
            # Rate limit (3 requests/second without API key)
            time.sleep(0.34)
            
        except pymongo.errors.DuplicateKeyError:
            skipped_count += 1
            continue
        except Exception as e:
            error_count += 1
            print(f"\n⚠️  Error with PMID {pmid}: {e}")
            continue
    
    # Sonuç raporu
    print(f"\n✓ Saved: {saved_count} papers")
    print(f"  - With full text: {full_text_count}")
    print(f"  - Skipped (duplicate): {skipped_count}")
    print(f"  - Errors: {error_count}")
    
    return saved_count


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Ana fonksiyon - tüm download işlemleri"""
    
    print("="*70)
    print("🏥 MediCrew PubMed → MongoDB Downloader")
    print("="*70)
    print()
    
    # Email kontrolü
    if Entrez.email == "your_email@example.com":
        print("❌ HATA: Lütfen Entrez.email'i kendi email adresinizle değiştirin!")
        print("   Script'in üst kısmındaki satırı düzenleyin:")
        print("   Entrez.email = 'sizin_email@example.com'")
        print()
        exit(1)
    
    # MongoDB bağlantısı
    collection = connect_mongodb()
    print()
    
    total_saved = 0
    
    # ================================================================
    # 1. KARDIYOLOJI PAPERS
    # ================================================================
    print("📕 CARDIOLOGY PAPERS")
    print("-"*70)
    
    cardio_queries = [
        'heart failure treatment guidelines',
        'coronary artery disease management',
        'hypertension therapy'
    ]
    
    for query in cardio_queries:
        count = download_to_mongodb(collection, query, 'cardiology', max_results=300)
        total_saved += count
        print()
    
    # ================================================================
    # 2. ENDOKRİNOLOJİ PAPERS
    # ================================================================
    print("\n📗 ENDOCRINOLOGY PAPERS")
    print("-"*70)
    
    endo_queries = [
        'type 2 diabetes management',
        'HbA1c control strategies',
        'insulin therapy protocols'
    ]
    
    for query in endo_queries:
        count = download_to_mongodb(collection, query, 'endocrinology', max_results=300)
        total_saved += count
        print()
    
    # ================================================================
    # 3. COMBINED PAPERS
    # ================================================================
    print("\n📘 COMBINED PAPERS (Cardiology + Endocrinology)")
    print("-"*70)
    
    combined_queries = [
        'diabetes cardiovascular disease',
        'diabetic cardiomyopathy'
    ]
    
    for query in combined_queries:
        count = download_to_mongodb(collection, query, 'combined', max_results=200)
        total_saved += count
        print()
    
    # ================================================================
    # FINAL STATISTICS
    # ================================================================
    print("\n" + "="*70)
    print("📊 FINAL STATISTICS")
    print("="*70)
    
    # Toplam sayılar
    total_count = collection.count_documents({})
    full_text_count = collection.count_documents({'full_text': {'$ne': None}})
    
    print(f"Total papers in MongoDB: {total_count}")
    print(f"Papers with full text: {full_text_count} ({full_text_count/total_count*100:.1f}%)")
    print(f"Papers with abstract only: {total_count - full_text_count}")
    
    # Domain dağılımı
    print("\nDomain breakdown:")
    for domain in ['cardiology', 'endocrinology', 'combined']:
        count = collection.count_documents({'domain': domain})
        print(f"  - {domain.capitalize()}: {count}")
    
    # Yıl dağılımı
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
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"📍 Database: {DB_NAME}")
    print(f"📍 Collection: {COLLECTION_NAME}")
    print("\n💡 Next Steps:")
    print("   1. MongoDB Compass ile veriyi görüntüleyin")
    print("   2. Embeddings oluşturun (OpenAI API)")
    print("   3. Pinecone'a yükleyin")
    print("\n🚀 Ready for Pinecone upload!")


if __name__ == "__main__":
    main()