"""
PubMed to MongoDB Downloader Core Module
Contains only core download functionality
"""

from Bio import Entrez
import time
import pymongo
from tqdm import tqdm
from datetime import datetime, timezone
from pymongo.errors import DuplicateKeyError
import os
import xml.etree.ElementTree as ET
import re

# ============================================================
# PUBMED API FUNCTIONS
# ============================================================

def search_pubmed(query, max_results=400, email=None, api_key=None):
    """Search PubMed and return list of PMIDs"""
    
    # Set Entrez credentials if provided
    if email:
        Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    
    # Rate limit based on whether API key exists
    rate_limit = 0.1 if api_key else 0.34
    
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
        print(f"⚠️ Error fetching PMID {pmid}: {e}")
        return None

# ============================================================
# MONGODB OPERATIONS
# ============================================================

def download_to_mongodb(collection, query, domain, max_results=400, email=None, api_key=None):
    """Download from PubMed and save directly to MongoDB"""
    
    # Search
    pmids = search_pubmed(query, max_results, email, api_key)
    
    # Rate limit based on whether API key exists
    rate_limit = 0.1 if api_key else 0.34
    
    if not pmids:
        print("⚠️ No results found")
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
            time.sleep(rate_limit)
            
        except DuplicateKeyError:
            skipped_count += 1
            continue
        except Exception as e:
            error_count += 1
            print(f"\n⚠️ Error with PMID {pmid}: {e}")
            continue
    
    # Summary report
    print(f"\n✅ Saved: {saved_count} papers")
    print(f"  - With full text: {full_text_count}")
    print(f"  - Skipped (duplicate): {skipped_count}")
    print(f"  - Errors: {error_count}")
    
    return saved_count