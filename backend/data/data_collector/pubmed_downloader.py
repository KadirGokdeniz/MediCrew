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
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

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
            if len(author_list) <= 4:
                # 4 or less show anyone
                authors = []
                for author in author_list:
                    name = f"{author.get('LastName', '')} {author.get('ForeName', author.get('Initials', ''))}".strip()
                    authors.append(name)
                authors_str = ', '.join(authors)
                if len(author_list) > 1:
                    authors_str += f" ({len(author_list)} authors)"
            else:
                # 5+ authors
                first_three = []
                for i in range(3):
                    author = author_list[i]
                    name = f"{author.get('LastName', '')} {author.get('ForeName', author.get('Initials', ''))}".strip()
                    first_three.append(name)
                
                last_author = author_list[-1]
                last_name = f"{last_author.get('LastName', '')} {last_author.get('ForeName', last_author.get('Initials', ''))}".strip()
                
                authors_str = f"{', '.join(first_three)}, ..., {last_name} ({len(author_list)} authors)"
        
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
    print("🏥 MediCrew PubMed → MongoDB Downloader (Cardiology Focus)")
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
    # CARDIOLOGY PAPERS ONLY (EKG included)
    # ================================================================
    print("🔴 CARDIOLOGY PAPERS (Including EKG)")
    print("-"*70)
    
    cardio_queries = [
    # Major society guidelines
    '("Practice Guideline"[Publication Type] OR "guideline"[Title]) AND ("Cardiology"[MeSH] OR "cardiovascular"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Specific guideline organizations
    '("AHA"[Title/Abstract] OR "ACC"[Title/Abstract] OR "American Heart Association"[Title/Abstract] OR "American College of Cardiology"[Title/Abstract]) AND ("guideline"[Title/Abstract] OR "recommendation"[Title/Abstract]) AND 2018:2025[pdat]',
    
    '("ESC"[Title/Abstract] OR "European Society of Cardiology"[Title/Abstract]) AND ("guideline"[Title/Abstract] OR "recommendation"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Indications & contraindications
    '("indication"[Title/Abstract] OR "indications"[Title/Abstract]) AND ("cardiovascular"[MeSH] OR "cardiology"[Title/Abstract]) AND 2018:2025[pdat]',
    
    '("contraindication"[Title/Abstract] OR "contraindicated"[Title/Abstract]) AND ("cardiovascular drug"[Title/Abstract] OR "cardiac"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Treatment algorithms
    '("treatment algorithm"[Title/Abstract] OR "management protocol"[Title/Abstract]) AND ("heart"[MeSH] OR "cardiovascular"[MeSH]) AND 2018:2025[pdat]',
    
    # Clinical decision rules
    '("clinical decision rule"[Title/Abstract] OR "risk score"[Title/Abstract] OR "risk stratification"[Title/Abstract]) AND ("cardiovascular"[MeSH] OR "cardiac"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Takotsubo
    '("Takotsubo Cardiomyopathy"[MeSH] OR "takotsubo"[Title/Abstract] OR "stress cardiomyopathy"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Valvular diseases 
    '("Aortic Valve Stenosis"[MeSH] OR "aortic stenosis"[Title/Abstract]) AND ("asymptomatic"[Title/Abstract] OR "monitoring"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Aortic pathologies
    '("Aortic Dissection"[MeSH] OR "aortic dissection"[Title/Abstract]) AND ("emergency"[Title/Abstract] OR "acute management"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiomyopathies
    '("Cardiomyopathy, Hypertrophic"[MeSH] OR "hypertrophic cardiomyopathy"[Title/Abstract] OR "HCM"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Myocarditis
    '("Myocarditis"[MeSH] OR "myocarditis"[Title/Abstract]) AND ("cardiac MRI"[Title/Abstract] OR "CMR"[Title/Abstract] OR "diagnosis"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Endocarditis
    '("Endocarditis"[MeSH] OR "infective endocarditis"[Title/Abstract]) AND ("diagnosis"[Title/Abstract] OR "Duke criteria"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Pericardial diseases
    '("Pericarditis"[MeSH] OR "pericarditis"[Title/Abstract] OR "pericardial effusion"[Title/Abstract]) AND 2018:2025[pdat]',

    # PCI vs alternatives
    '("Percutaneous Coronary Intervention"[MeSH] OR "PCI"[Title/Abstract]) AND ("stable coronary disease"[Title/Abstract] OR "stable angina"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # CABG
    '("Coronary Artery Bypass"[MeSH] OR "CABG"[Title/Abstract]) AND ("multivessel disease"[Title/Abstract] OR "left main"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # TAVR vs SAVR 
    '("Transcatheter Aortic Valve Replacement"[MeSH] OR "TAVR"[Title/Abstract] OR "TAVI"[Title/Abstract]) AND ("surgical"[Title/Abstract] OR "low risk"[Title/Abstract] OR "intermediate risk"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # CRT
    '("Cardiac Resynchronization Therapy"[MeSH] OR "CRT"[Title/Abstract] OR "biventricular pacing"[Title/Abstract]) AND ("indication"[Title/Abstract] OR "guideline"[Title/Abstract] OR "primary prevention"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # ICD
    '("Defibrillators, Implantable"[MeSH] OR "ICD"[Title/Abstract] OR "implantable cardioverter defibrillator"[Title/Abstract]) AND ("primary prevention"[Title/Abstract] OR "indication"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Mechanical circulatory support
    '("Heart-Assist Devices"[MeSH] OR "mechanical circulatory support"[Title/Abstract] OR "ECMO"[Title/Abstract] OR "Impella"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Catheter ablation
    '("Catheter Ablation"[MeSH] OR "ablation"[Title/Abstract]) AND ("atrial fibrillation"[Title/Abstract] OR "ventricular tachycardia"[Title/Abstract]) AND 2018:2025[pdat]',

    # Anticoagulation
    '("Anticoagulants"[MeSH] OR "anticoagulation"[Title/Abstract]) AND ("atrial fibrillation"[MeSH] OR "atrial fibrillation"[Title/Abstract]) AND ("bleeding"[Title/Abstract] OR "hemorrhage"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # DOACs vs Warfarin
    '("direct oral anticoagulants"[Title/Abstract] OR "DOAC"[Title/Abstract] OR "NOAC"[Title/Abstract]) AND ("warfarin"[Title/Abstract] OR "vitamin K antagonist"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # ACE-I vs ARB
    '("Angiotensin-Converting Enzyme Inhibitors"[MeSH] OR "ACE inhibitor"[Title/Abstract]) AND ("heart failure"[MeSH] OR "hypertension"[MeSH]) AND 2018:2025[pdat]',
    
    '("Angiotensin Receptor Antagonists"[MeSH] OR "ARB"[Title/Abstract]) AND ("heart failure"[MeSH] OR "hypertension"[MeSH]) AND 2018:2025[pdat]',
    
    # Comparison
    '("ACE inhibitor"[Title/Abstract] OR "angiotensin converting enzyme inhibitor"[Title/Abstract]) AND ("ARB"[Title/Abstract] OR "angiotensin receptor blocker"[Title/Abstract]) AND ("comparison"[Title/Abstract] OR "versus"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Beta-blockers
    '("Adrenergic beta-Antagonists"[MeSH] OR "beta blocker"[Title/Abstract]) AND ("heart failure"[Title/Abstract] OR "post myocardial infarction"[Title/Abstract] OR "elderly"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Antiplatelet
    '("dual antiplatelet therapy"[Title/Abstract] OR "DAPT"[Title/Abstract]) AND ("duration"[Title/Abstract] OR "drug eluting stent"[Title/Abstract] OR "DES"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # PCSK9 inhibitors (yoktu)
    '("PCSK9 Inhibitors"[MeSH] OR "PCSK9 inhibitor"[Title/Abstract] OR "evolocumab"[Title/Abstract] OR "alirocumab"[Title/Abstract]) AND ("secondary prevention"[Title/Abstract] OR "cardiovascular"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Lipid lowering in CKD
    '("Hydroxymethylglutaryl-CoA Reductase Inhibitors"[MeSH] OR "statin"[Title/Abstract]) AND ("chronic kidney disease"[MeSH] OR "CKD"[Title/Abstract] OR "dialysis"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Pregnancy contraindications
    '("Pregnancy"[MeSH] OR "pregnant"[Title/Abstract]) AND ("cardiovascular agents"[MeSH] OR "cardiac medication"[Title/Abstract]) AND ("contraindication"[Title/Abstract] OR "teratogenic"[Title/Abstract] OR "safety"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Perioperative management
    '("Perioperative Period"[MeSH] OR "perioperative"[Title/Abstract]) AND ("anticoagulation"[Title/Abstract] OR "antiplatelet"[Title/Abstract]) AND ("non-cardiac surgery"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Troponin (başarılıydı, devam)
    '("Troponin"[MeSH] OR "troponin"[Title/Abstract]) AND ("myocardial infarction"[MeSH] OR "acute coronary syndrome"[Title/Abstract]) AND ("interpretation"[Title/Abstract] OR "dynamics"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # BNP/NT-proBNP
    '("Natriuretic Peptide, Brain"[MeSH] OR "BNP"[Title/Abstract] OR "NT-proBNP"[Title/Abstract]) AND ("heart failure"[MeSH] OR "diagnosis"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # ECG interpretation (daha detaylı)
    '("Electrocardiography"[MeSH] OR "ECG interpretation"[Title/Abstract]) AND ("atrial flutter"[Title/Abstract] OR "atrial fibrillation"[Title/Abstract] OR "ventricular tachycardia"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Advanced echo
    '("Echocardiography"[MeSH] OR "echocardiography"[Title/Abstract]) AND ("strain imaging"[Title/Abstract] OR "speckle tracking"[Title/Abstract] OR "3D echo"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiac MRI specific findings
    '("Magnetic Resonance Imaging"[MeSH] OR "cardiac MRI"[Title/Abstract] OR "CMR"[Title/Abstract]) AND ("myocarditis"[Title/Abstract] OR "cardiomyopathy"[Title/Abstract] OR "Lake Louise"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Stress testing
    '("Exercise Test"[MeSH] OR "stress test"[Title/Abstract] OR "exercise testing"[Title/Abstract]) AND ("coronary artery disease"[MeSH] OR "ischemia"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiac CT
    '("Computed Tomography Angiography"[MeSH] OR "coronary CT"[Title/Abstract] OR "CCTA"[Title/Abstract]) AND 2018:2025[pdat]',

    # STEMI management
    '("ST Elevation Myocardial Infarction"[MeSH] OR "STEMI"[Title/Abstract]) AND ("management"[Title/Abstract] OR "treatment"[Title/Abstract] OR "first 24 hours"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Acute aortic syndrome 
    '("aortic dissection"[Title/Abstract] OR "aortic syndrome"[Title/Abstract]) AND ("emergency"[Title/Abstract] OR "acute management"[Title/Abstract] OR "diagnosis"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiogenic shock 
    '("Shock, Cardiogenic"[MeSH] OR "cardiogenic shock"[Title/Abstract]) AND ("stabilization"[Title/Abstract] OR "mechanical support"[Title/Abstract] OR "management"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiac arrest
    '("Heart Arrest"[MeSH] OR "cardiac arrest"[Title/Abstract]) AND ("resuscitation"[Title/Abstract] OR "post-arrest care"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Unstable arrhythmias
    '("ventricular tachycardia"[Title/Abstract] OR "ventricular fibrillation"[Title/Abstract]) AND ("unstable"[Title/Abstract] OR "acute management"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Hypertensive emergency
    '("Hypertension"[MeSH] OR "hypertensive emergency"[Title/Abstract] OR "hypertensive crisis"[Title/Abstract]) AND ("management"[Title/Abstract] OR "treatment"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Acute pulmonary edema
    '("Pulmonary Edema"[MeSH] OR "acute pulmonary edema"[Title/Abstract]) AND ("management"[Title/Abstract] OR "treatment"[Title/Abstract]) AND 2018:2025[pdat]',

    # Pregnancy and cardiovascular disease
    '("Pregnancy"[MeSH] OR "pregnant women"[Title/Abstract]) AND ("heart failure"[Title/Abstract] OR "cardiovascular disease"[Title/Abstract] OR "arrhythmia"[Title/Abstract]) AND ("management"[Title/Abstract] OR "treatment"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # CKD and cardiovascular disease
    '("Renal Insufficiency, Chronic"[MeSH] OR "chronic kidney disease"[Title/Abstract] OR "CKD"[Title/Abstract]) AND ("cardiovascular"[Title/Abstract] OR "heart disease"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Elderly/geriatric cardiology
    '("Aged"[MeSH] OR "elderly"[Title/Abstract] OR "geriatric"[Title/Abstract]) AND ("cardiovascular disease"[MeSH] OR "heart failure"[Title/Abstract] OR "hypertension"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Diabetes and cardiovascular
    '("Diabetes Mellitus"[MeSH] OR "diabetes"[Title/Abstract]) AND ("cardiovascular disease"[Title/Abstract] OR "coronary artery disease"[Title/Abstract] OR "heart failure"[Title/Abstract]) AND 2018:2025[pdat]',

    # Secondary prevention
    '("Secondary Prevention"[MeSH] OR "secondary prevention"[Title/Abstract]) AND ("cardiovascular disease"[MeSH] OR "myocardial infarction"[Title/Abstract]) AND ("lifestyle"[Title/Abstract] OR "exercise"[Title/Abstract] OR "diet"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Familial hypercholesterolemia 
    '("Hyperlipoproteinemia Type II"[MeSH] OR "familial hypercholesterolemia"[Title/Abstract]) AND ("screening"[Title/Abstract] OR "cascade testing"[Title/Abstract] OR "genetic"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Cardiac rehabilitation
    '("Cardiac Rehabilitation"[MeSH] OR "cardiac rehabilitation"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Risk stratification scores
    '("risk stratification"[Title/Abstract] OR "risk score"[Title/Abstract]) AND ("cardiovascular"[Title/Abstract] OR "cardiac"[Title/Abstract]) AND ("CHA2DS2-VASc"[Title/Abstract] OR "GRACE"[Title/Abstract] OR "TIMI"[Title/Abstract] OR "HAS-BLED"[Title/Abstract]) AND 2018:2025[pdat]',

    # Post-MI follow-up
    '("Myocardial Infarction"[MeSH] OR "myocardial infarction"[Title/Abstract]) AND ("follow-up"[Title/Abstract] OR "post-discharge"[Title/Abstract] OR "monitoring"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Syncope evaluation
    '("Syncope"[MeSH] OR "syncope"[Title/Abstract]) AND ("risk stratification"[Title/Abstract] OR "evaluation"[Title/Abstract] OR "ECG"[Title/Abstract]) AND 2018:2025[pdat]',
    
    # Heart failure monitoring
    '("Heart Failure"[MeSH] OR "heart failure"[Title/Abstract]) AND ("monitoring"[Title/Abstract] OR "follow-up"[Title/Abstract] OR "surveillance"[Title/Abstract]) AND 2018:2025[pdat]',
    ]
    
    for query in cardio_queries:
        count = download_to_mongodb(collection, query, 'cardiology', max_results=40)
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
    
    print("\n✅ DOWNLOAD COMPLETE!")
    print(f"📁 Database: {DB_NAME}")
    print(f"📁 Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()