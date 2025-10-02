"""
Hybrid Chunking Strategy for Medical Papers
- XML-aware: Parses section structure if available
- Sentence-based fallback: For non-XML or abstract-only papers
- Intelligent overlap and boundary preservation
"""

import pymongo
from typing import List, Dict, Optional
import re
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET

# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medicrew"
SOURCE_COLLECTION = "pubmed_papers"
CHUNKS_COLLECTION = "paper_chunks"

# Chunking parameters
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_TOKENS = 100  # merge short chunks


# ============================================================
# UTILITIES
# ============================================================

def estimate_tokens(text: str) -> int:
    """Token sayısını tahmin et (~4 char = 1 token)"""
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """Cümlelere ayır"""
    # Medical abbreviations handeling
    text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '.<ABBREV>', text)  # U.S.A.
    text = re.sub(r'Dr\.', 'Dr<DOT>', text)
    text = re.sub(r'vs\.', 'vs<DOT>', text)
    text = re.sub(r'i\.e\.', 'i<DOT>e<DOT>', text)
    text = re.sub(r'e\.g\.', 'e<DOT>g<DOT>', text)
    
    # Split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore
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
        
        # Çok uzun cümle - zorla böl
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Uzun cümleyi kelime bazında böl
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
            
            # Overlap ekle
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
    """Text XML formatında mı kontrol et"""
    xml_indicators = ['<sec', '<title>', '<p>', '</sec>', 'xmlns']
    return any(indicator in text for indicator in xml_indicators)


def clean_xml_tags(text: str) -> str:
    """XML tag'lerini temizle, sadece text'i al"""
    # Remove XML tags but keep text
    text = re.sub(r'<[^>]+>', ' ', text)
    # Multiple spaces to single
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_xml_sections(xml_text: str) -> List[Dict[str, str]]:
    """
    XML'den section'ları parse et
    Returns: [{'title': 'Introduction', 'content': '...'}, ...]
    """
    try:
        # XML namespace'leri kaldır (basitleştirme)
        xml_text = re.sub(r'xmlns[^=]*="[^"]*"', '', xml_text)
        xml_text = re.sub(r'<\?xml[^>]*\?>', '', xml_text)
        
        # Parse
        root = ET.fromstring(xml_text)
        
        sections = []
        
        # Find all <sec> tags
        for sec in root.findall('.//sec'):
            title_elem = sec.find('title')
            title = title_elem.text if title_elem is not None else 'Untitled Section'
            
            # Get all text from this section (excluding nested sections)
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
                       max_tokens: int = 512) -> List[Dict[str, str]]:
    """
    Section'ları chunk'la
    Her section için title + content chunk'ları oluştur
    """
    chunks = []
    
    for section in sections:
        title = section['title']
        content = section['content']
        
        # Section çok kısa - tek chunk
        section_tokens = estimate_tokens(title + content)
        if section_tokens <= max_tokens:
            chunks.append({
                'section': title,
                'text': f"{title}\n\n{content}"
            })
        else:
            # Section'ı sentence-based chunk'la
            text_chunks = chunk_text_by_sentences(content, max_tokens)
            for i, chunk_text in enumerate(text_chunks):
                # İlk chunk'a title ekle
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
    1. Abstract → Always single chunk
    2. Full text → Try XML parsing, fallback to sentence-based
    """
    chunks = []
    pmid = paper.get('pmid')
    title = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    
    # ========================================
    # CHUNK 0: Abstract (Always)
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
            'created_at': datetime.utcnow()
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
            # XML parsing başarılı
            section_chunks = chunk_xml_sections(sections, MAX_CHUNK_TOKENS)
            
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
                    'created_at': datetime.utcnow()
                }
                chunks.append(chunk)
            
            return chunks
    
    # Strategy B: Fallback to sentence-based
    # XML parsing başarısız veya XML değil
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
            'created_at': datetime.utcnow()
        }
        chunks.append(chunk)
    
    return chunks


# ============================================================
# MONGODB OPERATIONS
# ============================================================

def connect_mongodb():
    """MongoDB bağlantısı"""
    print("🔌 Connecting to MongoDB...")
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
        print(f"❌ MongoDB connection failed: {e}")
        exit(1)


def setup_chunks_collection(chunks_collection):
    """Chunks collection setup"""
    # Mevcut chunk'ları temizle
    existing = chunks_collection.count_documents({})
    if existing > 0:
        print(f"\n⚠️  {existing} chunk zaten var")
        choice = input("Silip yeniden oluştur? (y/n): ")
        if choice.lower() == 'y':
            chunks_collection.delete_many({})
            print("✓ Eski chunk'lar silindi")
        else:
            print("İşlem iptal edildi")
            return False
    
    # Index'leri oluştur
    print("\n📑 Creating indexes...")
    chunks_collection.create_index('pmid')
    chunks_collection.create_index([('pmid', 1), ('chunk_index', 1)], unique=True)
    chunks_collection.create_index('chunk_type')
    chunks_collection.create_index('domain')
    chunks_collection.create_index('has_xml_structure')
    chunks_collection.create_index('embedded')
    chunks_collection.create_index('synced_to_pinecone')
    print("✓ Indexes created")
    
    return True


def process_all_papers(source_collection, chunks_collection):
    """Tüm paper'ları chunk'la"""
    
    total_papers = source_collection.count_documents({})
    print(f"\n📄 Processing {total_papers} papers...")
    
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
            
            # İstatistik topla
            if len(chunks) == 1:
                stats['abstract_only'] += 1
            else:
                stats['with_full_text'] += 1
                
                # XML mi sentence-based mi?
                if chunks[1].get('has_xml_structure'):
                    stats['xml_parsed'] += 1
                else:
                    stats['sentence_based'] += 1
    
    return stats


def show_statistics(chunks_collection, stats):
    """Detaylı istatistikler"""
    print("\n" + "="*70)
    print("📊 CHUNKING STATISTICS")
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
    
    # Chunk type dağılımı
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
    print(f"\n📄 Sample chunks:")
    
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
        print(f"\n2. XML-Parsed Chunk:")
        print(f"   PMID: {xml_sample['pmid']}")
        print(f"   Section: {xml_sample['section']}")
        print(f"   Tokens: {xml_sample['token_count']}")
        print(f"   Preview: {xml_sample['text'][:150]}...")
    
    # Sentence sample
    sent_sample = chunks_collection.find_one({'chunk_type': 'full_text'})
    if sent_sample:
        print(f"\n3. Sentence-Based Chunk:")
        print(f"   PMID: {sent_sample['pmid']}")
        print(f"   Tokens: {sent_sample['token_count']}")
        print(f"   Preview: {sent_sample['text'][:150]}...")


# ============================================================
# MAIN
# ============================================================

def main():
    """Ana fonksiyon"""
    
    print("="*70)
    print("🔪 Hybrid Medical Paper Chunking")
    print("="*70)
    print(f"\nStrategy:")
    print(f"  1. XML-aware section parsing (if available)")
    print(f"  2. Sentence-based fallback")
    print(f"  Max chunk size: {MAX_CHUNK_TOKENS} tokens")
    print(f"  Overlap: {CHUNK_OVERLAP} tokens")
    
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
    print("✅ CHUNKING COMPLETE")
    print("="*70)
    print(f"\n📍 Database: {DB_NAME}")
    print(f"📍 Collection: {CHUNKS_COLLECTION}")
    print("\n💡 Next Steps:")
    print("   1. Create embeddings (OpenAI API)")
    print("   2. Upload to Pinecone")
    print("\n🚀 Ready for embedding generation!")


if __name__ == "__main__":
    main()