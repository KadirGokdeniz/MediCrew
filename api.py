"""
PubMed API Service with MongoDB
FastAPI + MongoDB ile çalışan PubMed veri çekme servisi
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from Bio import Entrez
import time
from datetime import datetime
import pymongo
from pymongo.errors import DuplicateKeyError

# ============================================================
# CONFIGURATION
# ============================================================

# FastAPI uygulaması
app = FastAPI(
    title="MediCrew PubMed API",
    description="PubMed papers için REST API + MongoDB servisi",
    version="2.0.0"
)

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
COLLECTION_NAME = "pubmed_papers"

# Entrez email
DEFAULT_EMAIL = ""
Entrez.email = DEFAULT_EMAIL

# Global MongoDB collection
collection = None

# ============================================================
# MONGODB CONNECTION
# ============================================================

def get_mongodb_collection():
    """Get MongoDB collection'ı """
    global collection
    if collection is None:
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
            # Test connection
            client.server_info()
            
            # Create index on pmid for faster lookups
            collection.create_index("pmid", unique=True)
            
            return collection
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"MongoDB connection failed: {str(e)}"
            )
    return collection

# ============================================================
# MODELS
# ============================================================

class SearchRequest(BaseModel):
    query: str
    max_results: int = 100
    domain: str = "general"
    email: Optional[EmailStr] = None
    api_key: Optional[str] = None
    save_to_db: bool = True

class PaperResponse(BaseModel):
    pmid: str
    pmc_id: Optional[str]
    title: str
    abstract: str
    full_text: Optional[str]
    journal: str
    year: Optional[int]
    authors: str
    pubmed_url: str
    pmc_url: Optional[str]
    domain: str
    has_full_text: bool
    pmc_available: bool
    downloaded_at: datetime
    synced_to_pinecone: bool

class SearchResponse(BaseModel):
    query: str
    domain: str
    total_found: int
    saved_to_db: int
    skipped_duplicate: int
    errors: int
    papers: List[PaperResponse]
    execution_time: float

class DBStats(BaseModel):
    total_papers: int
    with_full_text: int
    by_domain: Dict[str, int]
    recent_years: List[Dict[str, int]]

# ============================================================
# PUBMED FUNCTIONS
# ============================================================

def set_email(email: str):
    """Entrez email'i ayarla"""
    Entrez.email = email

def set_api_key(api_key: str):
    """Entrez API key'i ayarla"""
    if api_key:
        Entrez.api_key = api_key

def search_pubmed(query: str, max_results: int = 100):
    """PubMed'de arama yap"""
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
        return results['IdList']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubMed search failed: {str(e)}")

def get_pmc_id(pmid: str):
    """PMID'den PMC ID bul"""
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

def fetch_full_text_from_pmc(pmc_id: str):
    """PMC'den full text çek"""
    try:
        handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
        )
        xml_content = handle.read()
        handle.close()
        
        if xml_content and len(xml_content) > 1000:
            text = xml_content.decode('utf-8', errors='ignore')
            return text[:50000] if len(text) > 50000 else text
        return None
    except:
        return None

def fetch_paper_details(pmid: str, domain: str = "general", query: str = ""):
    """Tek paper'ın tüm detaylarını çek ve MongoDB formatında döndür"""
    try:
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
        
        # Basic info
        pmid_str = str(medline['PMID'])
        title = str(article.get('ArticleTitle', 'No Title'))
        
        # Abstract
        abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_parts, list):
            abstract = ' '.join([str(part) for part in abstract_parts])
        else:
            abstract = str(abstract_parts) if abstract_parts else ''
        
        # Journal & Year
        journal = article.get('Journal', {}).get('Title', 'Unknown')
        pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        year_str = pub_date.get('Year', None)
        year = int(year_str) if year_str and year_str.isdigit() else None
        
        # Authors
        author_list = article.get('AuthorList', [])
        authors = []
        for author in author_list[:5]:
            last = author.get('LastName', '')
            init = author.get('Initials', '')
            if last:
                authors.append(f"{last} {init}".strip())
        authors_str = ', '.join(authors) if authors else 'Unknown'
        
        # URLs
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
        
        # Full text
        full_text = None
        pmc_id = get_pmc_id(pmid_str)
        pmc_url = None
        
        if pmc_id:
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            full_text = fetch_full_text_from_pmc(pmc_id)
            time.sleep(0.2)  # Rate limit
        
        # MongoDB formatında döndür
        return {
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
            'domain': domain,
            'downloaded_at': datetime.utcnow(),
            'synced_to_pinecone': False,
            'metadata': {
                'has_full_text': full_text is not None,
                'pmc_available': pmc_id is not None,
                'query': query
            }
        }
        
    except Exception as e:
        return None

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Servis başlarken MongoDB'ye bağlan"""
    get_mongodb_collection()
    print("✅ MongoDB connection established")

@app.get("/")
async def root():
    """API ana sayfası"""
    return {
        "name": "MediCrew PubMed API + MongoDB",
        "version": "2.0.0",
        "endpoints": {
            "search": "/search (POST) - Arama yap ve MongoDB'ye kaydet",
            "paper": "/paper/{pmid} (GET) - Tek paper getir/kaydet",
            "batch": "/batch (POST) - Toplu paper getir/kaydet",
            "stats": "/stats (GET) - MongoDB istatistikleri",
            "cardiology": "/domains/cardiology (GET)",
            "endocrinology": "/domains/endocrinology (GET)",
            "combined": "/domains/combined (GET)",
            "health": "/health (GET)"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    coll = get_mongodb_collection()
    total_docs = coll.count_documents({})
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "mongodb_connected": True,
        "total_papers": total_docs,
        "email_configured": Entrez.email != DEFAULT_EMAIL
    }

@app.get("/stats", response_model=DBStats)
async def get_statistics():
    """MongoDB istatistiklerini getir"""
    coll = get_mongodb_collection()
    
    total = coll.count_documents({})
    with_full_text = coll.count_documents({'full_text': {'$ne': None}})
    
    # Domain breakdown
    by_domain = {}
    for domain in ['cardiology', 'endocrinology', 'combined', 'general']:
        count = coll.count_documents({'domain': domain})
        if count > 0:
            by_domain[domain] = count
    
    # Year distribution
    pipeline = [
        {'$match': {'year': {'$ne': None}}},
        {'$group': {'_id': '$year', 'count': {'$sum': 1}}},
        {'$sort': {'_id': -1}},
        {'$limit': 5}
    ]
    year_stats = list(coll.aggregate(pipeline))
    recent_years = [{'year': stat['_id'], 'count': stat['count']} for stat in year_stats]
    
    return DBStats(
        total_papers=total,
        with_full_text=with_full_text,
        by_domain=by_domain,
        recent_years=recent_years
    )

@app.post("/search", response_model=SearchResponse)
async def search_and_save(request: SearchRequest):
    """
    PubMed'de arama yap ve MongoDB'ye kaydet
    
    - **query**: Arama sorgusu
    - **max_results**: Maksimum sonuç sayısı (default: 100)
    - **domain**: Veri domain'i (cardiology, endocrinology, combined, general)
    - **email**: NCBI Entrez email (opsiyonel)
    - **api_key**: NCBI API key (opsiyonel)
    - **save_to_db**: MongoDB'ye kaydet (default: True)
    """
    start_time = time.time()
    
    # Email ve API key ayarla
    if request.email:
        set_email(request.email)
    if request.api_key:
        set_api_key(request.api_key)
    
    # MongoDB collection
    coll = get_mongodb_collection()
    
    # Arama yap
    pmids = search_pubmed(request.query, request.max_results)
    
    if not pmids:
        return SearchResponse(
            query=request.query,
            domain=request.domain,
            total_found=0,
            saved_to_db=0,
            skipped_duplicate=0,
            errors=0,
            papers=[],
            execution_time=time.time() - start_time
        )
    
    # Paper detaylarını çek ve kaydet
    papers = []
    saved_count = 0
    skipped_count = 0
    error_count = 0
    
    for pmid in pmids:
        try:
            # MongoDB'de var mı kontrol et
            if request.save_to_db:
                existing = coll.find_one({'pmid': str(pmid)})
                if existing:
                    skipped_count += 1
                    # Var olan paper'ı listeye ekle
                    papers.append({
                        **existing,
                        'has_full_text': existing['metadata']['has_full_text'],
                        'pmc_available': existing['metadata']['pmc_available']
                    })
                    continue
            
            # Paper detaylarını çek
            paper = fetch_paper_details(pmid, request.domain, request.query)
            
            if not paper:
                error_count += 1
                continue
            
            # MongoDB'ye kaydet
            if request.save_to_db:
                try:
                    coll.insert_one(paper.copy())
                    saved_count += 1
                except DuplicateKeyError:
                    skipped_count += 1
            
            # Response için format
            paper_response = {
                **paper,
                'has_full_text': paper['metadata']['has_full_text'],
                'pmc_available': paper['metadata']['pmc_available']
            }
            papers.append(paper_response)
            
            # Rate limit
            time.sleep(0.34)
            
        except Exception as e:
            error_count += 1
            continue
    
    execution_time = time.time() - start_time
    
    return SearchResponse(
        query=request.query,
        domain=request.domain,
        total_found=len(pmids),
        saved_to_db=saved_count,
        skipped_duplicate=skipped_count,
        errors=error_count,
        papers=papers,
        execution_time=execution_time
    )

@app.get("/paper/{pmid}")
async def get_or_fetch_paper(
    pmid: str,
    domain: str = Query("general", description="Paper domain"),
    email: Optional[str] = Query(None, description="NCBI Entrez email"),
    api_key: Optional[str] = Query(None, description="NCBI API key"),
    force_refresh: bool = Query(False, description="Yeniden çek")
):
    """
    Paper'ı MongoDB'den getir, yoksa PubMed'den çek ve kaydet
    
    - **pmid**: PubMed ID
    - **force_refresh**: True ise yeniden çek
    """
    if email:
        set_email(email)
    if api_key:
        set_api_key(api_key)
    
    coll = get_mongodb_collection()
    
    # MongoDB'de var mı kontrol et
    if not force_refresh:
        existing = coll.find_one({'pmid': pmid})
        if existing:
            return {
                **existing,
                'source': 'mongodb',
                'has_full_text': existing['metadata']['has_full_text'],
                'pmc_available': existing['metadata']['pmc_available']
            }
    
    # PubMed'den çek
    paper = fetch_paper_details(pmid, domain)
    
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper not found: {pmid}")
    
    # MongoDB'ye kaydet
    try:
        coll.insert_one(paper.copy())
    except DuplicateKeyError:
        # Zaten var, update et
        coll.replace_one({'pmid': pmid}, paper)
    
    return {
        **paper,
        'source': 'pubmed',
        'has_full_text': paper['metadata']['has_full_text'],
        'pmc_available': paper['metadata']['pmc_available']
    }

@app.post("/batch")
async def batch_fetch_and_save(
    pmids: List[str],
    domain: str = Query("general"),
    email: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Birden fazla PMID için toplu veri çekme ve MongoDB'ye kaydetme
    """
    if email:
        set_email(email)
    if api_key:
        set_api_key(api_key)
    
    if len(pmids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 PMIDs allowed per request")
    
    coll = get_mongodb_collection()
    
    papers = []
    saved = 0
    skipped = 0
    
    for pmid in pmids:
        # Var mı kontrol et
        existing = coll.find_one({'pmid': str(pmid)})
        if existing:
            papers.append(existing)
            skipped += 1
            continue
        
        # Çek
        paper = fetch_paper_details(pmid, domain)
        if paper:
            # Kaydet
            try:
                coll.insert_one(paper.copy())
                saved += 1
            except DuplicateKeyError:
                skipped += 1
            
            papers.append(paper)
        
        time.sleep(0.34)
    
    return {
        "requested": len(pmids),
        "fetched": len(papers),
        "saved_to_db": saved,
        "skipped_duplicate": skipped,
        "papers": papers
    }

@app.get("/domains/cardiology")
async def cardiology_papers(
    max_results: int = Query(300, ge=1, le=500),
    email: Optional[str] = None
):
    """Kardiyoloji papers - önceden tanımlı sorgular"""
    queries = [
        "heart failure treatment guidelines",
        "coronary artery disease management",
        "hypertension therapy"
    ]
    
    all_stats = {
        'domain': 'cardiology',
        'queries': [],
        'total_saved': 0,
        'total_skipped': 0
    }
    
    for query in queries:
        request = SearchRequest(
            query=query,
            max_results=max_results // len(queries),
            domain='cardiology',
            email=email
        )
        result = await search_and_save(request)
        all_stats['queries'].append({
            'query': query,
            'saved': result.saved_to_db,
            'skipped': result.skipped_duplicate
        })
        all_stats['total_saved'] += result.saved_to_db
        all_stats['total_skipped'] += result.skipped_duplicate
    
    return all_stats

@app.get("/domains/endocrinology")
async def endocrinology_papers(
    max_results: int = Query(300, ge=1, le=500),
    email: Optional[str] = None
):
    """Endokrinoloji papers - önceden tanımlı sorgular"""
    queries = [
        "type 2 diabetes management",
        "HbA1c control strategies",
        "insulin therapy protocols"
    ]
    
    all_stats = {
        'domain': 'endocrinology',
        'queries': [],
        'total_saved': 0,
        'total_skipped': 0
    }
    
    for query in queries:
        request = SearchRequest(
            query=query,
            max_results=max_results // len(queries),
            domain='endocrinology',
            email=email
        )
        result = await search_and_save(request)
        all_stats['queries'].append({
            'query': query,
            'saved': result.saved_to_db,
            'skipped': result.skipped_duplicate
        })
        all_stats['total_saved'] += result.saved_to_db
        all_stats['total_skipped'] += result.skipped_duplicate
    
    return all_stats

@app.get("/domains/combined")
async def combined_papers(
    max_results: int = Query(200, ge=1, le=500),
    email: Optional[str] = None
):
    """Combined papers (Cardiology + Endocrinology)"""
    queries = [
        "diabetes cardiovascular disease",
        "diabetic cardiomyopathy"
    ]
    
    all_stats = {
        'domain': 'combined',
        'queries': [],
        'total_saved': 0,
        'total_skipped': 0
    }
    
    for query in queries:
        request = SearchRequest(
            query=query,
            max_results=max_results // len(queries),
            domain='combined',
            email=email
        )
        result = await search_and_save(request)
        all_stats['queries'].append({
            'query': query,
            'saved': result.saved_to_db,
            'skipped': result.skipped_duplicate
        })
        all_stats['total_saved'] += result.saved_to_db
        all_stats['total_skipped'] += result.skipped_duplicate
    
    return all_stats

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🚀 MediCrew PubMed API + MongoDB Server")
    print("="*70)
    print()
    print("📡 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔬 ReDoc: http://localhost:8000/redoc")
    print()
    print("💾 MongoDB: mongodb://localhost:27017/medicrew")
    print("📦 Collection: pubmed_papers")
    print()
    print("⚠️  IMPORTANT: Set your email in requests or code!")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)