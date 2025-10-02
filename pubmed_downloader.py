"""
PubMed to MongoDB Downloader
Downloads PubMed papers and saves to MongoDB
"""

from Bio import Entrez
import time
import pymongo
from tqdm import tqdm
from datetime import datetime
from pymongo.errors import DuplicateKeyError

# ============================================================
# CONFIGURATION
# ============================================================

# NCBI Entrez Configuration
Entrez.email = "kadirqokdeniz@hotmail.com"  # CHANGE THIS!
# Entrez.api_key = "YOUR_API_KEY"  # Optional - faster downloads

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
COLLECTION_NAME = "pubmed_papers"

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
        collection.drop_indexes()  
        
        # Test connection
        client.server_info()
        print(f"✅ Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        
        # Create unique index on pmid
        collection.create_index("pmid", unique=True)
        
        # Show existing document count
        existing_count = collection.count_documents({})
        print(f"📊 Existing documents: {existing_count}")
        
        return collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Solutions:")
        print("   1. Is MongoDB running? Terminal: mongod")
        print("   2. Check MongoDB connection string")
        exit(1)

# ============================================================
# PUBMED API FUNCTIONS
# ============================================================

def search_pubmed(query, max_results=500):
    """Search PubMed and return list of PMIDs"""
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
        print(f"✅ Found {len(pmids)} articles")
        return pmids
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def get_pmc_id(pmid):
    """Get PMC ID from PMID (for full text access)"""
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
    """Fetch full text from PMC (Open Access only)"""
    try:
        handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
        )
        xml_content = handle.read()
        handle.close()
        
        # Convert XML to text (simplified)
        if xml_content and len(xml_content) > 1000:
            text = xml_content.decode('utf-8', errors='ignore')
            # Limit to 50k characters (MongoDB document size limit)
            return text[:50000] if len(text) > 50000 else text
        return None
    except Exception as e:
        return None


def fetch_paper_details(pmid):
    """Fetch all details for a single paper"""
    try:
        # Get basic info from PubMed
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
        for author in author_list[:5]:  # First 5 authors
            last = author.get('LastName', '')
            init = author.get('Initials', '')
            if last:
                authors.append(f"{last} {init}".strip())
        authors_str = ', '.join(authors) if authors else 'Unknown'
        
        # URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
        
        # Try to get full text from PMC
        full_text = None
        pmc_id = get_pmc_id(pmid_str)
        pmc_url = None
        
        if pmc_id:
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            full_text = fetch_full_text_from_pmc(pmc_id)
            # Extra wait for rate limiting
            time.sleep(0.2)
        
        # Create paper object
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
            'synced_to_pinecone': False,
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
    """Download from PubMed and save directly to MongoDB"""
    
    # Search
    pmids = search_pubmed(query, max_results)
    
    if not pmids:
        print("⚠️  No results found")
        return 0
    
    saved_count = 0
    skipped_count = 0
    error_count = 0
    full_text_count = 0
    
    print(f"📥 Downloading {len(pmids)} papers...")
    
    # Process with progress bar
    for pmid in tqdm(pmids, desc="Processing"):
        try:
            # Check for duplicates
            if collection.find_one({'pmid': str(pmid)}):
                skipped_count += 1
                continue
            
            # Fetch paper details
            paper = fetch_paper_details(pmid)
            
            if not paper:
                error_count += 1
                continue
            
            # Add domain and query info
            paper['domain'] = domain
            paper['metadata']['query'] = query
            
            # Save to MongoDB
            collection.insert_one(paper)
            saved_count += 1
            
            # Count full text papers
            if paper.get('full_text'):
                full_text_count += 1
            
            # Rate limit (3 requests/second without API key)
            time.sleep(0.34)
            
        except DuplicateKeyError:
            skipped_count += 1
            continue
        except Exception as e:
            error_count += 1
            print(f"\n⚠️  Error with PMID {pmid}: {e}")
            continue
    
    # Summary report
    print(f"\n✅ Saved: {saved_count} papers")
    print(f"  - With full text: {full_text_count}")
    print(f"  - Skipped (duplicate): {skipped_count}")
    print(f"  - Errors: {error_count}")
    
    return saved_count

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function - all download operations"""
    
    print("="*70)
    print("🏥 MediCrew PubMed → MongoDB Downloader")
    print("="*70)
    print()
    
    # Email check
    if Entrez.email == "your_email@example.com":
        print("❌ ERROR: Please set your email address!")
        print("   Edit the line at the top of the script:")
        print("   Entrez.email = 'your_email@example.com'")
        print()
        exit(1)
    
    # MongoDB connection
    collection = connect_mongodb()
    print()
    
    total_saved = 0
    
    # ================================================================
    # 1. CARDIOLOGY PAPERS
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
    # 2. ENDOCRINOLOGY PAPERS
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
    
    # Total counts
    total_count = collection.count_documents({})
    full_text_count = collection.count_documents({'full_text': {'$ne': None}})
    
    print(f"Total papers in MongoDB: {total_count}")
    print(f"Papers with full text: {full_text_count} ({full_text_count/total_count*100:.1f}%)")
    print(f"Papers with abstract only: {total_count - full_text_count}")
    
    # Domain breakdown
    print("\nDomain breakdown:")
    for domain in ['cardiology', 'endocrinology', 'combined']:
        count = collection.count_documents({'domain': domain})
        print(f"  - {domain.capitalize()}: {count}")
    
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
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"📁 Database: {DB_NAME}")
    print(f"📁 Collection: {COLLECTION_NAME}")
    print("\n💡 Next Steps:")
    print("   1. View data with MongoDB Compass")
    print("   2. Create embeddings (OpenAI API)")
    print("   3. Upload to Pinecone")
    print("\n🚀 Ready for API service!")


if __name__ == "__main__":
    main()