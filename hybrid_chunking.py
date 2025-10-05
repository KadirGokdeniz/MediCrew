"""
Hybrid Chunking Strategy for Medical Papers v2.0
- XML-aware: Parses section structure if available
- Sentence-based fallback: For non-XML or abstract-only papers
- Intelligent overlap for ALL chunking strategies
"""

import pymongo
from typing import List, Dict, Optional
import re
from tqdm import tqdm
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medicrew")
SOURCE_COLLECTION = "pubmed_papers"
CHUNKS_COLLECTION = "paper_chunks"

# Chunking parameters
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50  # Applied to ALL chunking strategies
MIN_CHUNK_TOKENS = 100


# ============================================================
# UTILITIES
# ============================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars = 1 token)"""
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling medical abbreviations"""
    # Handle medical abbreviations
    text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '.<ABBREV>', text)
    text = re.sub(r'Dr\.', 'Dr<DOT>', text)
    text = re.sub(r'vs\.', 'vs<DOT>', text)
    text = re.sub(r'i\.e\.', 'i<DOT>e<DOT>', text)
    text = re.sub(r'e\.g\.', 'e<DOT>g<DOT>', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore abbreviations
    sentences = [s.replace('<ABBREV>', '.').replace('<DOT>', '.') 
                 for s in sentences if s.strip()]
    
    return sentences


def chunk_text_by_sentences(text: str, max_tokens: int = 512, 
                            overlap: int = 50) -> List[str]:
    """Sentence-based chunking with overlap"""
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # Handle very long sentences - force split
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word)
                if temp_tokens + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            continue
        
        # Normal chunking
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Add overlap
            overlap_sentences = []
            overlap_tokens = 0
            for sent in reversed(current_chunk):
                sent_tokens = estimate_tokens(sent)
                if overlap_tokens + sent_tokens <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================
# XML PARSING
# ============================================================

def is_xml_content(text: str) -> bool:
    """Check if text is in XML format"""
    xml_indicators = ['<sec', '<title>', '<p>', '</sec>', 'xmlns']
    return any(indicator in text for indicator in xml_indicators)


def clean_xml_tags(text: str) -> str:
    """Remove XML tags, keep only text content"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_xml_sections(xml_text: str) -> List[Dict[str, str]]:
    """
    Parse sections from XML
    Returns: [{'title': 'Introduction', 'content': '...'}, ...]
    """
    try:
        # Remove XML namespaces
        xml_text = re.sub(r'xmlns[^=]*="[^"]*"', '', xml_text)
        xml_text = re.sub(r'<\?xml[^>]*\?>', '', xml_text)
        
        root = ET.fromstring(xml_text)
        
        sections = []
        
        # Find all <sec> tags
        for sec in root.findall('.//sec'):
            title_elem = sec.find('title')
            title = title_elem.text if title_elem is not None else 'Untitled Section'
            
            # Get all text from this section
            paragraphs = []
            for p in sec.findall('./p'):
                text = ET.tostring(p, encoding='unicode', method='text')
                if text.strip():
                    paragraphs.append(text.strip())
            
            content = ' '.join(paragraphs)
            
            if content.strip():
                sections.append({
                    'title': title.strip(),
                    'content': content.strip()
                })
        
        return sections
        
    except ET.ParseError:
        return []
    except Exception as e:
        return []


def chunk_xml_sections(sections: List[Dict[str, str]], 
                       max_tokens: int = 512,
                       overlap: int = 50) -> List[Dict[str, str]]:
    """
    Chunk sections with overlap support (UPDATED)
    Creates title + content chunks with overlap between chunks within sections
    """
    chunks = []
    
    for section in sections:
        title = section['title']
        content = section['content']
        
        # Section is short - single chunk
        section_tokens = estimate_tokens(title + content)
        if section_tokens <= max_tokens:
            chunks.append({
                'section': title,
                'text': f"{title}\n\n{content}"
            })
        else:
            # Chunk section with sentence-based approach WITH OVERLAP
            text_chunks = chunk_text_by_sentences(content, max_tokens, overlap)
            for i, chunk_text in enumerate(text_chunks):
                # Add title to each chunk
                if i == 0:
                    final_text = f"{title}\n\n{chunk_text}"
                else:
                    final_text = f"{title} (continued)\n\n{chunk_text}"
                
                chunks.append({
                    'section': title,
                    'text': final_text
                })
    
    return chunks


# ============================================================
# MAIN CHUNKING LOGIC
# ============================================================

def create_chunks_from_paper(paper: Dict) -> List[Dict]:
    """
    Hybrid chunking strategy:
    1. Abstract -> Always single chunk (no overlap needed)
    2. Full text (XML) -> Section-based with overlap
    3. Full text (non-XML) -> Sentence-based with overlap
    """
    chunks = []
    pmid = paper.get('pmid')
    title = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    
    # ========================================
    # CHUNK 0: Abstract (Always single chunk)
    # ========================================
    if title or abstract:
        abstract_text = f"{title}\n\n{abstract}".strip()
        
        chunk = {
            'pmid': pmid,
            'chunk_type': 'abstract',
            'chunk_index': 0,
            'section': 'Abstract',
            'text': abstract_text,
            'token_count': estimate_tokens(abstract_text),
            
            # Metadata
            'title': title,
            'journal': paper.get('journal'),
            'year': paper.get('year'),
            'authors': paper.get('authors'),
            'domain': paper.get('domain'),
            'pubmed_url': paper.get('pubmed_url'),
            'pmc_url': paper.get('pmc_url'),
            
            # Flags
            'has_xml_structure': False,
            'embedded': False,
            'embedding': None,
            'synced_to_pinecone': False,
            'created_at': datetime.now(timezone.utc)
        }
        chunks.append(chunk)
    
    # ========================================
    # CHUNKS 1-N: Full Text (If Available)
    # ========================================
    full_text = paper.get('full_text')
    if not full_text or not full_text.strip():
        return chunks
    
    # Strategy A: Try XML parsing
    if is_xml_content(full_text):
        sections = parse_xml_sections(full_text)
        
        if sections:
            # XML parsing successful - use overlap for consistency
            section_chunks = chunk_xml_sections(sections, MAX_CHUNK_TOKENS, CHUNK_OVERLAP)
            
            for idx, section_chunk in enumerate(section_chunks, start=1):
                chunk = {
                    'pmid': pmid,
                    'chunk_type': 'full_text_xml',
                    'chunk_index': idx,
                    'section': section_chunk['section'],
                    'text': section_chunk['text'],
                    'token_count': estimate_tokens(section_chunk['text']),
                    
                    # Metadata
                    'title': title,
                    'journal': paper.get('journal'),
                    'year': paper.get('year'),
                    'authors': paper.get('authors'),
                    'domain': paper.get('domain'),
                    'pubmed_url': paper.get('pubmed_url'),
                    'pmc_url': paper.get('pmc_url'),
                    
                    # Flags
                    'has_xml_structure': True,
                    'embedded': False,
                    'embedding': None,
                    'synced_to_pinecone': False,
                    'created_at': datetime.now(timezone.utc)
                }
                chunks.append(chunk)
            
            return chunks
    
    # Strategy B: Fallback to sentence-based with overlap
    clean_text = clean_xml_tags(full_text) if is_xml_content(full_text) else full_text
    text_chunks = chunk_text_by_sentences(clean_text, MAX_CHUNK_TOKENS, CHUNK_OVERLAP)
    
    for idx, chunk_text in enumerate(text_chunks, start=1):
        chunk = {
            'pmid': pmid,
            'chunk_type': 'full_text',
            'chunk_index': idx,
            'section': 'Full Text',
            'text': chunk_text,
            'token_count': estimate_tokens(chunk_text),
            
            # Metadata
            'title': title,
            'journal': paper.get('journal'),
            'year': paper.get('year'),
            'authors': paper.get('authors'),
            'domain': paper.get('domain'),
            'pubmed_url': paper.get('pubmed_url'),
            'pmc_url': paper.get('pmc_url'),
            
            # Flags
            'has_xml_structure': False,
            'embedded': False,
            'embedding': None,
            'synced_to_pinecone': False,
            'created_at': datetime.now(timezone.utc)
        }
        chunks.append(chunk)
    
    return chunks


# ============================================================
# MONGODB OPERATIONS
# ============================================================

def connect_mongodb():
    """Connect to MongoDB"""
    print("Connecting to MongoDB...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        source_collection = db[SOURCE_COLLECTION]
        chunks_collection = db[CHUNKS_COLLECTION]
        
        client.server_info()
        print(f"✓ Connected to MongoDB")
        print(f"  Source: {SOURCE_COLLECTION}")
        print(f"  Target: {CHUNKS_COLLECTION}")
        
        return source_collection, chunks_collection
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        exit(1)


def setup_chunks_collection(chunks_collection):
    """Setup chunks collection with defensive indexing"""
    existing = chunks_collection.count_documents({})
    if existing > 0:
        print(f"\n⚠️  Warning: {existing} chunks already exist")
        choice = input("Delete and recreate? (y/n): ")
        if choice.lower() == 'y':
            chunks_collection.delete_many({})
            print("✓ Old chunks deleted")
        else:
            print("Operation cancelled")
            return False
    
    # Create indexes defensively
    print("\nCreating indexes...")
    try:
        chunks_collection.create_index('pmid')
        chunks_collection.create_index([('pmid', 1), ('chunk_index', 1)], unique=True)
        chunks_collection.create_index('chunk_type')
        chunks_collection.create_index('domain')
        chunks_collection.create_index('has_xml_structure')
        chunks_collection.create_index('embedded')
        chunks_collection.create_index('synced_to_pinecone')
        print("✓ Indexes created")
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")
    
    return True


def process_all_papers(source_collection, chunks_collection):
    """Process all papers and create chunks"""
    
    total_papers = source_collection.count_documents({})
    print(f"\nProcessing {total_papers} papers...")
    
    papers = source_collection.find()
    
    # Statistics
    stats = {
        'total_papers': 0,
        'total_chunks': 0,
        'abstract_only': 0,
        'with_full_text': 0,
        'xml_parsed': 0,
        'sentence_based': 0
    }
    
    for paper in tqdm(papers, total=total_papers, desc="Chunking"):
        chunks = create_chunks_from_paper(paper)
        
        if chunks:
            chunks_collection.insert_many(chunks)
            stats['total_chunks'] += len(chunks)
            stats['total_papers'] += 1
            
            # Collect statistics
            if len(chunks) == 1:
                stats['abstract_only'] += 1
            else:
                stats['with_full_text'] += 1
                
                # XML or sentence-based?
                if chunks[1].get('has_xml_structure'):
                    stats['xml_parsed'] += 1
                else:
                    stats['sentence_based'] += 1
    
    return stats


def show_statistics(chunks_collection, stats):
    """Show detailed statistics"""
    print("\n" + "="*70)
    print("CHUNKING STATISTICS")
    print("="*70)
    
    print(f"\nPaper Processing:")
    print(f"  Total papers: {stats['total_papers']:,}")
    print(f"  Abstract only: {stats['abstract_only']:,} ({stats['abstract_only']/stats['total_papers']*100:.1f}%)")
    print(f"  With full text: {stats['with_full_text']:,} ({stats['with_full_text']/stats['total_papers']*100:.1f}%)")
    
    if stats['with_full_text'] > 0:
        print(f"\nFull-text chunking strategy:")
        print(f"  XML-parsed: {stats['xml_parsed']:,} ({stats['xml_parsed']/stats['with_full_text']*100:.1f}%)")
        print(f"  Sentence-based: {stats['sentence_based']:,} ({stats['sentence_based']/stats['with_full_text']*100:.1f}%)")
    
    print(f"\nChunk Output:")
    print(f"  Total chunks: {stats['total_chunks']:,}")
    print(f"  Avg chunks/paper: {stats['total_chunks']/stats['total_papers']:.1f}")
    
    # Chunk type distribution
    abstract_chunks = chunks_collection.count_documents({'chunk_type': 'abstract'})
    xml_chunks = chunks_collection.count_documents({'chunk_type': 'full_text_xml'})
    text_chunks = chunks_collection.count_documents({'chunk_type': 'full_text'})
    
    print(f"\nChunk types:")
    print(f"  Abstract: {abstract_chunks:,}")
    print(f"  Full text (XML): {xml_chunks:,}")
    print(f"  Full text (Sentence): {text_chunks:,}")
    
    # Token statistics
    pipeline = [
        {'$group': {
            '_id': None,
            'avg_tokens': {'$avg': '$token_count'},
            'min_tokens': {'$min': '$token_count'},
            'max_tokens': {'$max': '$token_count'}
        }}
    ]
    
    token_stats = list(chunks_collection.aggregate(pipeline))
    if token_stats:
        st = token_stats[0]
        print(f"\nToken statistics:")
        print(f"  Average: {st['avg_tokens']:.0f} tokens")
        print(f"  Min: {st['min_tokens']} tokens")
        print(f"  Max: {st['max_tokens']} tokens")
    
    # Sample chunks
    print(f"\nSample chunks:")
    
    # Abstract sample
    abstract_sample = chunks_collection.find_one({'chunk_type': 'abstract'})
    if abstract_sample:
        print(f"\n1. Abstract Chunk:")
        print(f"   PMID: {abstract_sample['pmid']}")
        print(f"   Tokens: {abstract_sample['token_count']}")
        print(f"   Preview: {abstract_sample['text'][:150]}...")
    
    # XML sample
    xml_sample = chunks_collection.find_one({'chunk_type': 'full_text_xml'})
    if xml_sample:
        print(f"\n2. XML-Parsed Chunk (with overlap):")
        print(f"   PMID: {xml_sample['pmid']}")
        print(f"   Section: {xml_sample['section']}")
        print(f"   Tokens: {xml_sample['token_count']}")
        print(f"   Preview: {xml_sample['text'][:150]}...")
    
    # Sentence sample
    sent_sample = chunks_collection.find_one({'chunk_type': 'full_text'})
    if sent_sample:
        print(f"\n3. Sentence-Based Chunk (with overlap):")
        print(f"   PMID: {sent_sample['pmid']}")
        print(f"   Tokens: {sent_sample['token_count']}")
        print(f"   Preview: {sent_sample['text'][:150]}...")
    
    print(f"\n💡 Note: All full-text chunks now use {CHUNK_OVERLAP} token overlap")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main function"""
    
    print("="*70)
    print("Hybrid Medical Paper Chunking v2.0")
    print("="*70)
    print(f"\nStrategy:")
    print(f"  1. XML-aware section parsing (if available)")
    print(f"  2. Sentence-based fallback")
    print(f"  Max chunk size: {MAX_CHUNK_TOKENS} tokens")
    print(f"  Overlap: {CHUNK_OVERLAP} tokens (ALL strategies)")
    
    # MongoDB setup
    source_collection, chunks_collection = connect_mongodb()
    
    # Chunks collection setup
    if not setup_chunks_collection(chunks_collection):
        return
    
    # Process papers
    stats = process_all_papers(source_collection, chunks_collection)
    
    # Show statistics
    show_statistics(chunks_collection, stats)
    
    print("\n" + "="*70)
    print("CHUNKING COMPLETE ✓")
    print("="*70)
    print(f"\nDatabase: {DB_NAME}")
    print(f"Collection: {CHUNKS_COLLECTION}")
    print("\nNext Steps:")
    print("   1. Run pinecone_integration.py to create embeddings")
    print("   2. Build RAG API for querying")
    print("\n🚀 Ready for embedding generation!")


if __name__ == "__main__":
    main()