"""
PubMed to MongoDB Downloader v2.0
Enhanced version with:
- Environment variables (.env)
- Smart indexing
- Hybrid MeSH + Keyword queries
- Section-based truncation
- Better error handling
"""

from Bio import Entrez
import time
import pymongo
from tqdm import tqdm
from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import re

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# NCBI Entrez Configuration
Entrez.email = os.getenv("ENTREZ_EMAIL")
Entrez.api_key = os.getenv("ENTREZ_API_KEY")  # Optional but recommended

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medicrew")
COLLECTION_NAME = "pubmed_papers"

# Rate limiting
RATE_LIMIT = 0.34 if not Entrez.api_key else 0.1  # With API key: 10 req/sec

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Validate required environment variables"""
    if not Entrez.email:
        print("❌ ERROR: ENTREZ_EMAIL not set in .env file!")
        print("\n💡 Create a .env file with:")
        print("   ENTREZ_EMAIL=your_email@example.com")
        exit(1)
    
    if not Entrez.api_key:
        print("⚠️  WARNING: ENTREZ_API_KEY not set")
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
        collection.drop_index("idx_pmid_unique")
        
        # Test connection
        client.server_info()
        print(f"✅ Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        
        # Smart index creation (only if not exists)
        setup_indexes(collection)
        
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


def setup_indexes(collection):
    """Create indexes only if they don't exist"""
    existing_indexes = [idx['name'] for idx in collection.list_indexes()]
    
    indexes_created = 0
    
    # PMID unique index
    if 'pmid_1' not in existing_indexes:
        collection.create_index("pmid", unique=True)
        print("  ✓ Created pmid index (unique)")
        indexes_created += 1
    
    # Domain index
    if 'domain_1' not in existing_indexes:
        collection.create_index("domain")
        print("  ✓ Created domain index")
        indexes_created += 1
    
    # Year index
    if 'year_1' not in existing_indexes:
        collection.create_index("year")
        print("  ✓ Created year index")
        indexes_created += 1
    
    # Synced flag index
    if 'synced_to_pinecone_1' not in existing_indexes:
        collection.create_index("synced_to_pinecone")
        print("  ✓ Created synced_to_pinecone index")
        indexes_created += 1
    
    if indexes_created == 0:
        print("  ✓ All indexes already exist")

# ============================================================
# PUBMED API FUNCTIONS
# ============================================================

def search_pubmed(query, max_results=400):
    """Search PubMed and return list of PMIDs"""
    print(f"🔍 Searching: '{query[:80]}...' (max {max_results} results)")    
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


def extract_priority_sections(xml_content):
    """
    Extract priority sections from XML
    Priority: Abstract > Results > Methods > Conclusion > Introduction
    Skip: Discussion (too long, repetitive)
    """
    try:
        # Clean XML
        xml_content = re.sub(r'xmlns[^=]*="[^"]*"', '', xml_content)
        xml_content = re.sub(r'<\?xml[^>]*\?>', '', xml_content)
        
        root = ET.fromstring(xml_content)
        
        # Priority sections mapping
        priority_map = {
            'abstract': 10,
            'intro': 3,
            'introduction': 3,
            'methods': 5,
            'materials': 5,
            'results': 8,
            'conclusion': 7,
            'conclusions': 7,
            'discussion': 0  # Skip discussion
        }
        
        sections = []
        
        # Find all sections
        for sec in root.findall('.//sec'):
            title_elem = sec.find('title')
            title = title_elem.text.lower() if title_elem is not None else ''
            
            # Get priority
            priority = 0
            for key, val in priority_map.items():
                if key in title:
                    priority = val
                    break
            
            if priority == 0:  # Skip low priority sections
                continue
            
            # Extract text
            paragraphs = []
            for p in sec.findall('./p'):
                text = ET.tostring(p, encoding='unicode', method='text')
                if text.strip():
                    paragraphs.append(text.strip())
            
            content = ' '.join(paragraphs)
            
            if content.strip():
                sections.append({
                    'title': title_elem.text if title_elem is not None else 'Section',
                    'content': content,
                    'priority': priority
                })
        
        # Sort by priority
        sections.sort(key=lambda x: x['priority'], reverse=True)
        
        # Build text with limits
        result = []
        total_chars = 0
        max_chars = 150000  # 150K character limit
        
        for section in sections:
            section_text = f"\n\n## {section['title']}\n{section['content']}"
            if total_chars + len(section_text) <= max_chars:
                result.append(section_text)
                total_chars += len(section_text)
            else:
                break
        
        return ''.join(result) if result else None
        
    except Exception as e:
        return None


def fetch_full_text_from_pmc(pmc_id):
    """Fetch full text from PMC with section-based truncation"""
    try:
        handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
        )
        xml_content = handle.read()
        handle.close()
        
        if not xml_content or len(xml_content) < 1000:
            return None
        
        xml_text = xml_content.decode('utf-8', errors='ignore')
        
        # Strategy 1: Try section-based extraction
        extracted = extract_priority_sections(xml_text)
        
        if extracted:
            return extracted
        
        # Strategy 2: Fallback - simple truncation
        # Remove XML tags
        clean_text = re.sub(r'<[^>]+>', ' ', xml_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Truncate to 100K
        return clean_text[:100000] if len(clean_text) > 100000 else clean_text
        
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
        
        # Authors - First + Last + Count
        author_list = article.get('AuthorList', [])
        authors_str = 'Unknown'
        
        if author_list:
            first_author = author_list[0]
            first_name = f"{first_author.get('LastName', '')} {first_author.get('Initials', '')}".strip()
            
            if len(author_list) > 1:
                last_author = author_list[-1]
                last_name = f"{last_author.get('LastName', '')} {last_author.get('Initials', '')}".strip()
                authors_str = f"{first_name}, ..., {last_name} ({len(author_list)} authors)"
            else:
                authors_str = first_name
        
        # URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
        
        # Try to get full text from PMC
        full_text = None
        pmc_id = get_pmc_id(pmid_str)
        pmc_url = None
        
        if pmc_id:
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            full_text = fetch_full_text_from_pmc(pmc_id)
            time.sleep(0.2)  # Extra wait for PMC
        
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
            'downloaded_at': datetime.now(timezone.utc),
            'synced_to_pinecone': False,
            'metadata': {
                'has_full_text': full_text is not None,
                'pmc_available': pmc_id is not None,
                'full_text_length': len(full_text) if full_text else 0
            }
        }
        
        return paper
        
    except Exception as e:
        print(f"⚠️  Error fetching PMID {pmid}: {e}")
        return None

# ============================================================
# MONGODB OPERATIONS
# ============================================================

def download_to_mongodb(collection, query, domain, max_results=400):
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
            
            # Rate limit
            time.sleep(RATE_LIMIT)
            
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
    print("🏥 MediCrew PubMed → MongoDB Downloader v2.0")
    print("="*70)
    print()
    
    # Validate configuration
    validate_config()
    print()
    
    # MongoDB connection
    collection = connect_mongodb()
    print()
    
    total_saved = 0
    
    # ================================================================
    # 1. CARDIOLOGY PAPERS
    # ================================================================
    print("🔴 CARDIOLOGY PAPERS")
    print("-"*70)
    
    cardio_queries = [
        # Hybrid MeSH + Keyword, 6-7 year range
        '("Heart Failure"[MeSH] OR "heart failure"[Title/Abstract]) AND 2018:2024[pdat]',
        '("Myocardial Infarction"[MeSH] OR "heart attack"[Title/Abstract]) AND 2018:2024[pdat]',
        '("Hypertension"[MeSH] OR "high blood pressure"[Title/Abstract]) AND 2018:2024[pdat]',
    ]
    
    for query in cardio_queries:
        count = download_to_mongodb(collection, query, 'cardiology', max_results=400)
        total_saved += count
        print()
    
    # ================================================================
    # 2. ENDOCRINOLOGY PAPERS
    # ================================================================
    print("\n🔵 ENDOCRINOLOGY PAPERS")
    print("-"*70)
    
    endo_queries = [
        '("Diabetes Mellitus, Type 2"[MeSH] OR "type 2 diabetes"[Title/Abstract]) AND 2018:2024[pdat]',
        '("Glycated Hemoglobin"[MeSH] OR "HbA1c"[Title/Abstract]) AND 2018:2024[pdat]',
        '("Insulin"[MeSH] OR "insulin therapy"[Title/Abstract]) AND 2018:2024[pdat]',
    ]
    
    for query in endo_queries:
        count = download_to_mongodb(collection, query, 'endocrinology', max_results=400)
        total_saved += count
        print()
    
    # ================================================================
    # 3. COMBINED PAPERS
    # ================================================================
    print("\n🟣 COMBINED PAPERS (Cardiology + Endocrinology)")
    print("-"*70)
    
    combined_queries = [
        '("Diabetes Mellitus"[MeSH] AND "Cardiovascular Diseases"[MeSH]) AND 2018:2024[pdat]',
        '("Diabetic Cardiomyopathies"[MeSH] OR "diabetic cardiomyopathy"[Title/Abstract]) AND 2018:2024[pdat]',
    ]
    
    for query in combined_queries:
        count = download_to_mongodb(collection, query, 'combined', max_results=300)
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
    
    print(f"\nTotal papers in MongoDB: {total_count}")
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
    
    # Full text stats
    if full_text_count > 0:
        print("\nFull text length stats:")
        pipeline = [
            {'$match': {'metadata.full_text_length': {'$gt': 0}}},
            {'$group': {
                '_id': None,
                'avg_length': {'$avg': '$metadata.full_text_length'},
                'max_length': {'$max': '$metadata.full_text_length'}
            }}
        ]
        length_stats = list(collection.aggregate(pipeline))
        if length_stats:
            st = length_stats[0]
            print(f"  - Average length: {st['avg_length']:,.0f} characters")
            print(f"  - Max length: {st['max_length']:,.0f} characters")
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"📁 Database: {DB_NAME}")
    print(f"📁 Collection: {COLLECTION_NAME}")
    print("\n💡 Next Steps:")
    print("   1. Run hybrid_chunking.py to create chunks")
    print("   2. Run pinecone_integration.py to create embeddings")
    print("\n🚀 Ready for RAG system!")


if __name__ == "__main__":
    main()