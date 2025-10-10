"""
Enhanced Hybrid Chunker v4.0
- Abstract ve full-text chunk'ları ayrı
- Token-safe chunking
- Optimized for two-stage retrieval
"""

import pymongo
from typing import List, Dict, Optional, Tuple
import re
from tqdm import tqdm
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medicrew")
SOURCE_COLLECTION = "pubmed_papers"
CHUNKS_COLLECTION = "paper_chunks"

# Chunking parameters
MAX_CHUNK_TOKENS = 450
CHUNK_OVERLAP = 64
MIN_CHUNK_TOKENS = 50

# Section-aware chunking
MIN_SECTION_WORDS = 50
MAX_SECTION_WORDS = 450
CHUNK_TARGET_WORDS = 250
OVERLAP_WORDS = 60

class ResearchPaperChunker:
    """Token-safe research paper chunker"""
    
    def __init__(self):
        self.section_priority = {
            'abstract': 10, 'introduction': 9, 'methods': 7, 
            'results': 8, 'discussion': 6, 'conclusion': 7,
            'background': 5, 'materials': 7, 'statistics': 6
        }
    
    def count_words(self, text: str) -> int:
        return len(text.split()) if text else 0
    
    def estimate_real_tokens(self, text: str) -> int:
        """Daha gerçekçi token hesabı"""
        words = text.split()
        return int(len(words) * 1.3)  # Tıp metinleri için approximation
    
    def detect_section_type(self, title: str) -> str:
        title_lower = title.lower()
        for section_type in self.section_priority:
            if section_type in title_lower:
                return section_type
        return 'other'
    
    def truncate_abstract(self, abstract: str, max_words: int = 200) -> str:
        """Abstract'ı anlamlı şekilde kısalt"""
        words = abstract.split()
        if len(words) <= max_words:
            return abstract
        
        # İlk cümleleri koru (genelde özet niteliğinde)
        truncated = ' '.join(words[:max_words])
        
        # Son cümleyi tamamlamaya çalış
        if '.' not in truncated[-50:]:
            last_dot = truncated.rfind('.')
            if last_dot != -1:
                truncated = truncated[:last_dot+1]
        
        return truncated + " [truncated]" if len(words) > max_words else truncated

    def merge_small_sections(self, sections: List[Dict]) -> List[Dict]:
        if not sections:
            return []
            
        merged_sections = []
        current_merge = None
        
        for section in sections:
            word_count = self.count_words(section['content'])
            
            if word_count < MIN_SECTION_WORDS:
                if current_merge is None:
                    current_merge = section.copy()
                else:
                    current_merge['content'] += " " + section['content']
                    current_merge['title'] += f" & {section['title']}"
            else:
                if current_merge is not None:
                    merged_sections.append(current_merge)
                    current_merge = None
                merged_sections.append(section)
        
        if current_merge is not None:
            merged_sections.append(current_merge)
            
        return merged_sections
    
    def split_large_section(self, section: Dict, target_words: int, overlap_words: int) -> List[Dict]:
        content = section['content']
        words = content.split()
        chunks = []
        
        start = 0
        part_num = 1
        
        while start < len(words):
            end = start + target_words
            if end > len(words):
                end = len(words)
            
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            # Cümle sınırlarını korumaya çalış
            sentences = re.split(r'(?<=[.!?])\s+', chunk_content)
            if len(sentences) > 1 and len(sentences[-1].split()) < 15:
                adjusted_text = ' '.join(sentences[:-1])
                adjusted_words = adjusted_text.split()
                end = start + len(adjusted_words)
                chunk_content = adjusted_text
            
            chunk = {
                'title': f"{section['title']} - Part {part_num}",
                'content': chunk_content,
                'word_count': self.count_words(chunk_content),
                'is_subsection': True
            }
            chunks.append(chunk)
            
            start += (target_words - overlap_words)
            part_num += 1
            
            if part_num > 100 or start >= len(words):
                break
                
        return chunks
    
    def process_sections(self, sections: List[Dict]) -> List[Dict]:
        if not sections:
            return []
            
        merged_sections = self.merge_small_sections(sections)
        processed_chunks = []
        
        for section in merged_sections:
            word_count = self.count_words(section['content'])
            
            if word_count <= MAX_SECTION_WORDS:
                processed_chunks.append(section)
            else:
                sub_sections = self.split_large_section(section, CHUNK_TARGET_WORDS, OVERLAP_WORDS)
                processed_chunks.extend(sub_sections)
        
        return processed_chunks

# ============================================================
# XML PARSING AND UTILITIES
# ============================================================

def is_xml_content(text: str) -> bool:
    if not text:
        return False
    xml_indicators = ['<sec', '<title>', '<p>', '</sec>', 'xmlns']
    return any(indicator in text for indicator in xml_indicators)

def parse_xml_sections(xml_text: str) -> List[Dict[str, str]]:
    try:
        xml_text = re.sub(r'xmlns[^=]*="[^"]*"', '', xml_text)
        xml_text = re.sub(r'<\?xml[^>]*\?>', '', xml_text)
        
        root = ET.fromstring(xml_text)
        sections = []
        
        for sec in root.findall('.//sec'):
            title_elem = sec.find('title')
            title = title_elem.text if title_elem is not None else 'Untitled Section'
            
            paragraphs = []
            for p in sec.findall('./p'):
                text = ET.tostring(p, encoding='unicode', method='text')
                if text and text.strip():
                    paragraphs.append(text.strip())
            
            content = ' '.join(paragraphs)
            
            if content.strip():
                sections.append({
                    'title': title.strip(),
                    'content': content.strip()
                })
        
        return sections
        
    except Exception as e:
        print(f"XML parsing error: {e}")
        return []

def clean_xml_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    
    # Tıbbi kısaltmaları koru
    text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '.<ABBREV>', text)
    text = re.sub(r'Dr\.', 'Dr<DOT>', text)
    text = re.sub(r'vs\.', 'vs<DOT>', text)
    text = re.sub(r'i\.e\.', 'i<DOT>e<DOT>', text)
    text = re.sub(r'e\.g\.', 'e<DOT>g<DOT>', text)
    text = re.sub(r'et al\.', 'et al<DOT>', text)
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<ABBREV>', '.').replace('<DOT>', '.') 
                 for s in sentences if s and s.strip()]
    
    return sentences

def chunk_text_by_sentences(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    chunker = ResearchPaperChunker()
    
    for sentence in sentences:
        sentence_tokens = chunker.estimate_real_tokens(sentence)
        
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = chunker.estimate_real_tokens(word)
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
        
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            overlap_sentences = []
            overlap_tokens = 0
            for sent in reversed(current_chunk):
                sent_tokens = chunker.estimate_real_tokens(sent)
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
# MAIN CHUNKING LOGIC - SEPARATE ABSTRACT & FULL-TEXT
# ============================================================

def create_chunks_from_paper(paper: Dict) -> List[Dict]:
    """
    YENİ: Abstract ve full-text chunk'ları ayrı
    - Abstract chunk'lar: sadece abstract
    - Full-text chunk'lar: sadece section content
    """
    chunks = []
    pmid = paper.get('pmid')
    title = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    chunker = ResearchPaperChunker()
    
    # ========================================
    # 1. ABSTRACT CHUNK (AYRI - sadece abstract)
    # ========================================
    if abstract:
        # Abstract'ı optimize et (token limiti için)
        optimized_abstract = chunker.truncate_abstract(abstract)
        
        abstract_chunk = {
            'pmid': pmid,
            'chunk_type': 'abstract_only',
            'chunk_index': 0,
            'section': 'Abstract',
            'text': optimized_abstract,  # ✅ SADECE abstract
            'token_count': chunker.estimate_real_tokens(optimized_abstract),
            'word_count': chunker.count_words(optimized_abstract),
            
            # Metadata for filtering
            'title': title,
            'journal': paper.get('journal'),
            'year': paper.get('year'),
            'authors': paper.get('authors'),
            'domain': paper.get('domain'),
            'pubmed_url': paper.get('pubmed_url'),
            'pmc_url': paper.get('pmc_url'),
            
            # Processing flags
            'is_abstract': True,
            'has_full_text': paper.get('full_text') is not None,
            'abstract_truncated': len(abstract.split()) > 200,
            'embedded': False,
            'synced_to_pinecone': False,
            'created_at': datetime.now(timezone.utc)
        }
        chunks.append(abstract_chunk)
    
    # ========================================
    # 2. FULL-TEXT CHUNKS (ABSTRACT OLMADAN)
    # ========================================
    full_text = paper.get('full_text')
    if not full_text or not full_text.strip():
        return chunks
    
    chunk_index = 1  # Abstract'tan sonra başla
    
    # Strategy A: Advanced XML parsing
    if is_xml_content(full_text):
        sections = parse_xml_sections(full_text)
        
        if sections:
            try:
                advanced_chunker = ResearchPaperChunker()
                processed_sections = advanced_chunker.process_sections(sections)
                
                for section in processed_sections:
                    # ✅ SADECE section content, abstract YOK
                    chunk_text = section['content']
                    
                    chunk = {
                        'pmid': pmid,
                        'chunk_type': 'full_text_section',
                        'chunk_index': chunk_index,
                        'section': section['title'],
                        'text': chunk_text,
                        'token_count': chunker.estimate_real_tokens(chunk_text),
                        'word_count': chunker.count_words(chunk_text),
                        
                        # Metadata
                        'title': title,
                        'journal': paper.get('journal'),
                        'year': paper.get('year'),
                        'authors': paper.get('authors'),
                        'domain': paper.get('domain'),
                        'pubmed_url': paper.get('pubmed_url'),
                        'pmc_url': paper.get('pmc_url'),
                        
                        # Processing flags
                        'is_abstract': False,
                        'has_xml_structure': True,
                        'is_subsection': section.get('is_subsection', False),
                        'embedded': False,
                        'synced_to_pinecone': False,
                        'created_at': datetime.now(timezone.utc)
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                
                return chunks
                
            except Exception as e:
                print(f"Advanced chunking failed for PMID {pmid}, falling back: {e}")
    
    # Strategy B: Traditional fallback
    clean_text = clean_xml_tags(full_text) if is_xml_content(full_text) else full_text
    
    text_chunks = chunk_text_by_sentences(clean_text, MAX_CHUNK_TOKENS, CHUNK_OVERLAP)
    
    for chunk_text in text_chunks:
        # ✅ SADECE chunk content, abstract YOK
        chunk = {
            'pmid': pmid,
            'chunk_type': 'full_text_traditional',
            'chunk_index': chunk_index,
            'section': 'Full Text',
            'text': chunk_text,
            'token_count': chunker.estimate_real_tokens(chunk_text),
            'word_count': chunker.count_words(chunk_text),
            
            # Metadata
            'title': title,
            'journal': paper.get('journal'),
            'year': paper.get('year'),
            'authors': paper.get('authors'),
            'domain': paper.get('domain'),
            'pubmed_url': paper.get('pubmed_url'),
            'pmc_url': paper.get('pmc_url'),
            
            # Processing flags
            'is_abstract': False,
            'has_xml_structure': False,
            'embedded': False,
            'synced_to_pinecone': False,
            'created_at': datetime.now(timezone.utc)
        }
        chunks.append(chunk)
        chunk_index += 1
    
    return chunks

# ============================================================
# MONGODB OPERATIONS (Aynı)
# ============================================================

def connect_mongodb():
    print("🔌 Connecting to MongoDB...")
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        db = client[DB_NAME]
        source_collection = db[SOURCE_COLLECTION]
        chunks_collection = db[CHUNKS_COLLECTION]
        
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {DB_NAME}")
        print(f"   Source: {SOURCE_COLLECTION}")
        print(f"   Target: {CHUNKS_COLLECTION}")
        
        return source_collection, chunks_collection
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        exit(1)

def setup_chunks_collection(chunks_collection):
    existing_count = chunks_collection.count_documents({})
    
    if existing_count > 0:
        print(f"⚠️  Warning: {existing_count} chunks already exist")
        choice = input("Delete and recreate? (y/n): ")
        if choice.lower() == 'y':
            chunks_collection.delete_many({})
            print("✓ Old chunks deleted")
        else:
            print("Operation cancelled")
            return False
    
    print("\nCreating indexes...")
    try:
        chunks_collection.create_index('pmid', name='pmid_index')
        chunks_collection.create_index([('pmid', pymongo.ASCENDING), ('chunk_index', pymongo.ASCENDING)], 
                                     unique=True, name='pmid_chunk_index')
        chunks_collection.create_index('chunk_type', name='chunk_type_index')
        chunks_collection.create_index('is_abstract', name='is_abstract_index')  # ✅ YENİ INDEX
        chunks_collection.create_index('domain', name='domain_index')
        chunks_collection.create_index('embedded', name='embedded_index')
        chunks_collection.create_index('synced_to_pinecone', name='synced_index')
        print("✓ Indexes created successfully")
        
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")
    return True

def process_all_papers(source_collection, chunks_collection):
    total_papers = source_collection.count_documents({})
    print(f"\n📄 Processing {total_papers} research papers...")
    
    papers = source_collection.find()
    
    stats = {
        'total_papers': 0,
        'total_chunks': 0,
        'abstract_chunks': 0,
        'fulltext_chunks': 0,
        'with_abstract_only': 0,
        'with_full_text': 0,
        'failed_processing': 0
    }
    
    for paper in tqdm(papers, total=total_papers, desc="Chunking Papers"):
        try:
            chunks = create_chunks_from_paper(paper)
            
            if chunks:
                chunks_collection.insert_many(chunks)
                
                stats['total_chunks'] += len(chunks)
                stats['total_papers'] += 1
                
                # Yeni istatistikler
                abstract_chunks = [c for c in chunks if c['is_abstract']]
                fulltext_chunks = [c for c in chunks if not c['is_abstract']]
                
                stats['abstract_chunks'] += len(abstract_chunks)
                stats['fulltext_chunks'] += len(fulltext_chunks)
                
                if len(fulltext_chunks) > 0:
                    stats['with_full_text'] += 1
                else:
                    stats['with_abstract_only'] += 1
                    
        except Exception as e:
            print(f"\n⚠️  Error processing paper {paper.get('pmid', 'unknown')}: {e}")
            stats['failed_processing'] += 1
            continue
    
    return stats

def show_detailed_statistics(chunks_collection, stats):
    print("\n" + "="*70)
    print("📊 ENHANCED CHUNKING STATISTICS")
    print("="*70)
    
    print(f"\n📄 PAPER PROCESSING SUMMARY:")
    print(f"   Total papers processed: {stats['total_papers']:,}")
    print(f"   Abstract-only papers: {stats['with_abstract_only']:,} ({stats['with_abstract_only']/stats['total_papers']*100:.1f}%)")
    print(f"   Papers with full text: {stats['with_full_text']:,} ({stats['with_full_text']/stats['total_papers']*100:.1f}%)")
    
    print(f"\n📦 CHUNK OUTPUT ANALYSIS:")
    print(f"   Total chunks created: {stats['total_chunks']:,}")
    print(f"   Abstract chunks: {stats['abstract_chunks']:,} ({stats['abstract_chunks']/stats['total_chunks']*100:.1f}%)")
    print(f"   Full-text chunks: {stats['fulltext_chunks']:,} ({stats['fulltext_chunks']/stats['total_chunks']*100:.1f}%)")
    print(f"   Average chunks per paper: {stats['total_chunks']/stats['total_papers']:.1f}")
    
    # Token analysis
    pipeline = [
        {"$group": {
            "_id": "$is_abstract",
            "avg_tokens": {"$avg": "$token_count"},
            "max_tokens": {"$max": "$token_count"},
            "over_512": {"$sum": {"$cond": [{"$gt": ["$token_count", 512]}, 1, 0]}},
            "count": {"$sum": 1}
        }}
    ]
    
    token_stats = list(chunks_collection.aggregate(pipeline))
    
    print(f"\n📏 TOKEN ANALYSIS:")
    for stat in token_stats:
        chunk_type = "Abstract" if stat['_id'] else "Full-text"
        pct_over = (stat['over_512'] / stat['count']) * 100
        print(f"   {chunk_type}: avg={stat['avg_tokens']:.0f}, max={stat['max_tokens']}, over_512={pct_over:.1f}%")
    
    print(f"\n✅ CHUNKING COMPLETED!")
    print(f"   Ready for two-stage retrieval system")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("🔬 ENHANCED HYBRID CHUNKER v4.0 - SEPARATE ABSTRACT & FULL-TEXT")
    print("="*70)
    
    print("\n🎯 NEW FEATURES:")
    print("   ✓ Abstract and full-text chunks SEPARATED")
    print("   ✓ Token-safe chunking with real token estimation")  
    print("   ✓ Optimized for two-stage retrieval")
    print("   ✓ Abstract truncation for long abstracts")
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Max tokens per chunk: {MAX_CHUNK_TOKENS}")
    print(f"   Abstract max words: 200")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} tokens")
    
    source_collection, chunks_collection = connect_mongodb()
    source_count = source_collection.count_documents({})
    print(f"\n📊 SOURCE DATA: {source_count} papers in {SOURCE_COLLECTION}")
    
    if not setup_chunks_collection(chunks_collection):
        return
    
    print(f"\n🚀 STARTING ENHANCED CHUNKING PROCESS...")
    stats = process_all_papers(source_collection, chunks_collection)
    show_detailed_statistics(chunks_collection, stats)
    
    print("\n" + "="*70)
    print("✅ ENHANCED CHUNKING COMPLETED!")
    print("="*70)
    print(f"\n📁 Database: {DB_NAME}")
    print(f"📁 Collection: {CHUNKS_COLLECTION}")
    print(f"📦 Total chunks: {stats['total_chunks']:,}")
    print(f"🔍 Abstract chunks: {stats['abstract_chunks']:,}")
    print(f"📄 Full-text chunks: {stats['fulltext_chunks']:,}")

if __name__ == "__main__":
    main()