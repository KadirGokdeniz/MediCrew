"""
Medicrew - Hybrid Retrieval Pipeline (BM25 + Dense) with Query Expansion
BM25 ile exact term matching + PubMedBERT ile semantic search + Medical Dictionary Expansion
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pymongo
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ROUTER_INDEX_NAME = "medical-abstracts-router"
UNIFIED_INDEX_NAME = "medical-papers-unified"

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medicrew")
CHUNKS_COLLECTION = "paper_chunks"

# Model
EMBEDDING_MODEL = "neuml/pubmedbert-base-embeddings"

# Retrieval parameters
ABSTRACT_TOP_K = 10
FULLTEXT_TOP_K = 20
SCORE_THRESHOLD = 0.60

# Hybrid parameters
BM25_WEIGHT = 0.3  # BM25 ağırlığı
DENSE_WEIGHT = 0.7  # Dense (semantic) ağırlığı

BM25_TOP_K = 50  # BM25'ten kaç sonuç al (dense'den önce)

# Query Expansion
DICTIONARY_PATH = "dictionary_words.json"
EXPANSION_LIMIT = 3  # Maksimum expansion sayısı


# ============================================================
# HYBRID RETRIEVER WITH QUERY EXPANSION
# ============================================================

class HybridMedicrewRetriever:
    """BM25 + Dense hybrid retrieval with medical dictionary expansion"""
    
    def __init__(self, dictionary_path: str = DICTIONARY_PATH):
        print("🚀 Hybrid Medicrew Retriever başlatılıyor...")
        
        # 1. Medical Dictionary yükle - YENİ EKLENDİ
        print("📚 Medical dictionary yükleniyor...")
        self.medical_dict = self.load_medical_dictionary(dictionary_path)
        print(f"✅ Medical dictionary hazır: {len(self.medical_dict)} terim")
        
        # 2. Embedding model
        print("📦 Embedding model yükleniyor...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model hazır")
        
        # 3. Pinecone
        print("🔗 Pinecone bağlantısı kuruluyor...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.router_index = pc.Index(ROUTER_INDEX_NAME)
        self.unified_index = pc.Index(UNIFIED_INDEX_NAME)
        print("✅ Pinecone bağlantısı hazır")
        
        # 4. MongoDB
        print("📚 MongoDB bağlantısı kuruluyor...")
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        self.chunks_collection = db[CHUNKS_COLLECTION]
        print("✅ MongoDB bağlantısı hazır")
        
        # 5. BM25 için abstract corpus'u yükle
        print("🔍 BM25 index oluşturuluyor...")
        self._build_bm25_index()
        print("✅ BM25 hazır")
        
        print("✅ Hybrid Retriever hazır!\n")
    
    def load_medical_dictionary(self, dictionary_path: str) -> Dict:
        """
        Medical dictionary'yi JSON'dan yükle ve optimize et
        """
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                dictionary_data = json.load(f)
            
            # Dictionary'yi daha hızlı lookup için optimize et
            medical_dict = {}
            for entry in dictionary_data:
                abbr = entry['abbr']
                medical_dict[abbr] = {
                    'preferred': entry['preferred'],
                    'full_forms': entry['full_forms'],
                    'synonyms': entry['synonyms'],
                    'aliases': entry['aliases'],
                    'semantic_type': entry['semantic_type'],
                    'confidence': entry['confidence']
                }
            
            print(f"   📖 {len(medical_dict)} medical terim yüklendi")
            return medical_dict
            
        except Exception as e:
            print(f"❌ Dictionary yüklenemedi: {e}")
            return {}
    
    def expand_query_with_dict(self, query: str) -> str:
        """
        Query'yi medical dictionary ile akıllıca genişlet
        Strategy: Sadece high-confidence terimleri genişlet
        """
        if not self.medical_dict:
            return query
        
        original_query = query
        print(f"🔍 Original query: {query}")
        
        # Query'yi kelimelere ayır ve kısaltmaları bul
        words = query.split()
        expanded_terms = []
        expansion_occurred = False
        
        for word in words:
            # Noktalama işaretlerini temizle ve uppercase yap
            clean_word = re.sub(r'[^\w]', '', word.upper())
            
            # Dictionary'de var mı kontrol et
            if clean_word in self.medical_dict:
                entry = self.medical_dict[clean_word]
                
                # Sadece high confidence terimleri genişlet
                if entry['confidence'] in ['high', 'medium']:
                    expansion_occurred = True
                    
                    # Preferred term + en iyi synonym'leri ekle
                    expansions = [entry['preferred']] + entry['synonyms'][:1]  # 1 synonym
                    
                    # Expansion'ı ekle
                    expanded_terms.append(word)  # Orijinal kelimeyi koru
                    expanded_terms.extend(expansions)
                    
                    print(f"   📝 '{clean_word}' → {expansions}")
                else:
                    expanded_terms.append(word)
            else:
                expanded_terms.append(word)
        
        if expansion_occurred:
            # Orijinal query + expanded terms birleştir (duplicate'leri kaldır)
            all_terms = list(dict.fromkeys(expanded_terms))
            expanded_query = " ".join(all_terms)
            
            # Query uzunluğunu kontrol et (çok uzunsa kısalt)
            if len(expanded_query.split()) > len(original_query.split()) + EXPANSION_LIMIT:
                expanded_query = " ".join(expanded_terms[:len(original_query.split()) + EXPANSION_LIMIT])
            
            print(f"✅ Expanded query: {expanded_query}")
            return expanded_query
        else:
            print("   ℹ️  No expansion needed")
            return original_query
    
    def _build_bm25_index(self):
        """MongoDB'den abstract'ları al ve BM25 index oluştur"""
        
        # Tüm abstract'ları çek
        abstracts = list(self.chunks_collection.find(
            {'is_abstract': True},
            {'pmid': 1, 'text': 1, 'title': 1, '_id': 0}
        ))
        
        print(f"   {len(abstracts)} abstract yüklendi")
        
        # Corpus oluştur
        self.abstract_corpus = []
        self.abstract_metadata = []
        
        for abstract in abstracts:
            # Text + Title birleştir (daha iyi coverage için)
            combined_text = f"{abstract.get('title', '')} {abstract.get('text', '')}"
            
            # Tokenize (basit - whitespace split)
            tokens = combined_text.lower().split()
            
            self.abstract_corpus.append(tokens)
            self.abstract_metadata.append({
                'pmid': abstract['pmid'],
                'text': abstract.get('text', ''),
                'title': abstract.get('title', '')
            })
        
        # BM25 oluştur
        self.bm25 = BM25Okapi(self.abstract_corpus)
        
        print(f"   BM25 corpus: {len(self.abstract_corpus)} documents")
    
    def embed_query(self, query: str) -> List[float]:
        """Query'yi embed et"""
        embedding = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
    
    def search_bm25(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
        """BM25 ile keyword-based search"""
        
        # Query tokenize
        query_tokens = query.lower().split()
        
        # BM25 score'ları hesapla
        scores = self.bm25.get_scores(query_tokens)
        
        # Top-k index'leri al
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Sonuçları oluştur
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Sadece score > 0 olanlar
                results.append({
                    'pmid': self.abstract_metadata[idx]['pmid'],
                    'score': float(score),
                    'text': self.abstract_metadata[idx]['text'],
                    'title': self.abstract_metadata[idx]['title'],
                    'method': 'bm25'
                })
        
        return results
    
    def search_dense(self, query_embedding: List[float], top_k: int = ABSTRACT_TOP_K) -> List[Dict]:
        """Dense (semantic) search - Pinecone"""
        
        results = self.router_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format sonuçlar
        formatted = []
        for match in results.get('matches', []):
            formatted.append({
                'pmid': match['metadata']['pmid'],
                'score': match['score'],
                'text': match['metadata'].get('text_preview', ''),
                'title': match['metadata'].get('title', ''),
                'metadata': match['metadata'],
                'method': 'dense'
            })
        
        return formatted
    
    def hybrid_search_abstracts(self, query: str, query_embedding: List[float], 
                               top_k: int = ABSTRACT_TOP_K) -> List[Dict]:
        """
        Hybrid search: BM25 + Dense birleştir
        
        Returns:
            List of results with hybrid scores
        """
        
        # 1. BM25 search
        bm25_results = self.search_bm25(query, top_k=BM25_TOP_K)
        
        # 2. Dense search
        dense_results = self.search_dense(query_embedding, top_k=top_k)
        
        # 3. Score normalization
        # BM25 score'ları normalize et (0-1 range)
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            
            for result in bm25_results:
                result['normalized_score'] = (result['score'] - min_bm25) / bm25_range
        
        # Dense score'lar zaten 0-1 arası (cosine similarity)
        for result in dense_results:
            result['normalized_score'] = result['score']
        
        # 4. PMID bazında birleştir
        combined = {}
        
        # BM25 sonuçlarını ekle
        for result in bm25_results:
            pmid = result['pmid']
            combined[pmid] = {
                'pmid': pmid,
                'bm25_score': result['normalized_score'],
                'dense_score': 0.0,  # Henüz dense'den gelmedi
                'text': result['text'],
                'title': result['title'],
                'metadata': result.get('metadata', {})
            }
        
        # Dense sonuçlarını ekle/güncelle
        for result in dense_results:
            pmid = result['pmid']
            if pmid in combined:
                combined[pmid]['dense_score'] = result['normalized_score']
                # Metadata'yı dense'den al (daha complete)
                combined[pmid]['metadata'] = result['metadata']
            else:
                combined[pmid] = {
                    'pmid': pmid,
                    'bm25_score': 0.0,
                    'dense_score': result['normalized_score'],
                    'text': result['text'],
                    'title': result['title'],
                    'metadata': result['metadata']
                }
        
        # 5. Hybrid score hesapla
        for pmid, data in combined.items():
            hybrid_score = (
                BM25_WEIGHT * data['bm25_score'] + 
                DENSE_WEIGHT * data['dense_score']
            )
            data['hybrid_score'] = hybrid_score
            data['score'] = hybrid_score  # Ana score
        
        # 6. Hybrid score'a göre sırala
        results = list(combined.values())
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results[:top_k]
    
    def search_fulltext(self, query_embedding: List[float], relevant_pmids: List[str], 
                       top_k: int = FULLTEXT_TOP_K) -> Dict:
        """Full-text index'de ara"""
        
        results = self.unified_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={'pmid': {'$in': relevant_pmids}}
        )
        return results
    
    def should_search_fulltext(self, abstract_results: List[Dict]) -> Tuple[bool, str]:
        """Routing decision"""
        
        if not abstract_results or len(abstract_results) < 3:
            return True, "insufficient_abstracts"
        
        # Top-3 hybrid score ortalaması
        top_scores = [r['hybrid_score'] for r in abstract_results[:1]]
        avg_score = sum(top_scores) / len(top_scores)
        
        if avg_score < SCORE_THRESHOLD:
            return True, "low_confidence"
        
        return False, "sufficient"
    
    def retrieve(self, query: str, force_fulltext: bool = False) -> Dict:
        """
        Hybrid retrieval with query expansion - ana fonksiyon
        """
        
        # 1. QUERY EXPANSION - YENİ EKLENEN KISIM
        expanded_query = self.expand_query_with_dict(query)
        
        print(f"🔍 Query: {query}")
        if query != expanded_query:
            print(f"📝 Expanded: {expanded_query}")
        print()
        
        # 2. Query embedding (expanded query ile)
        print("📊 Embedding oluşturuluyor...")
        query_embedding = self.embed_query(expanded_query)
        
        # 3. Hybrid abstract search (expanded query ile)
        print(f"🔀 Hybrid search (BM25 + Dense) yapılıyor...")
        abstract_results = self.hybrid_search_abstracts(expanded_query, query_embedding, ABSTRACT_TOP_K)
        
        abstract_count = len(abstract_results)
        print(f"✅ {abstract_count} abstract bulundu (hybrid)")
        
        if abstract_count > 0:
            top = abstract_results[0]
            print(f"   En yüksek score: {top['hybrid_score']:.3f}")
            print(f"      BM25: {top['bm25_score']:.3f}")
            print(f"      Dense: {top['dense_score']:.3f}")
        
        # 4. Routing decision
        fulltext_results = []
        searched_fulltext = False
        
        if force_fulltext:
            routing_decision = "forced"
            need_fulltext = True
            print("\n🎯 Full-text araması zorunlu kılındı")
        else:
            need_fulltext, routing_decision = self.should_search_fulltext(abstract_results)
            
            if need_fulltext:
                print(f"\n🎯 Full-text'e dalış gerekli (sebep: {routing_decision})")
            else:
                print(f"\n✅ Abstract'lar yeterli (threshold: {SCORE_THRESHOLD})")
        
        # 5. Full-text search
        if need_fulltext and abstract_count > 0:
            relevant_pmids = [r['pmid'] for r in abstract_results[:5]]
            
            print(f"📚 Full-text index'de arama yapılıyor ({len(relevant_pmids)} makale)...")
            fulltext_response = self.search_fulltext(
                query_embedding, 
                relevant_pmids, 
                FULLTEXT_TOP_K
            )
            
            fulltext_results = fulltext_response.get('matches', [])
            fulltext_count = len(fulltext_results)
            print(f"✅ {fulltext_count} full-text chunk bulundu")
            
            if fulltext_count > 0:
                top_score = fulltext_results[0]['score']
                print(f"   En yüksek score: {top_score:.3f}")
            
            searched_fulltext = True
        
        # 6. Sonuçları döndür
        result = {
            'original_query': query,  # Orijinal query'yi sakla
            'expanded_query': expanded_query,
            'query_embedding': query_embedding,
            'abstract_results': abstract_results,
            'fulltext_results': fulltext_results,
            'routing_decision': routing_decision,
            'searched_fulltext': searched_fulltext,
            'abstract_count': abstract_count,
            'fulltext_count': len(fulltext_results),
            'retrieval_method': 'hybrid_with_expansion',  # Güncellendi
            'bm25_weight': BM25_WEIGHT,
            'dense_weight': DENSE_WEIGHT,
            'expansion_used': query != expanded_query,  # Expansion kullanıldı mı?
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Hybrid retrieval with expansion tamamlandı!")
        return result
    
    def format_results(self, results: Dict, include_full_text: bool = False) -> str:
        """Sonuçları formatla"""
        
        output = []
        output.append("="*70)
        output.append("MEDICREW HYBRID RETRIEVAL WITH QUERY EXPANSION")
        output.append("="*70)
        output.append(f"\nOriginal Query: {results['original_query']}")
        if results['expansion_used']:
            output.append(f"Expanded Query: {results['expanded_query']}")
        output.append(f"Method: {results['retrieval_method'].upper()}")
        output.append(f"  BM25 weight: {results['bm25_weight']}")
        output.append(f"  Dense weight: {results['dense_weight']}")
        output.append(f"  Query Expansion: {'YES' if results['expansion_used'] else 'NO'}")
        output.append(f"Routing: {results['routing_decision']}")
        output.append(f"Full-text searched: {results['searched_fulltext']}")
        
        # Abstract results
        output.append(f"\n{'='*70}")
        output.append(f"ABSTRACT RESULTS ({results['abstract_count']} found)")
        output.append("="*70)
        
        for i, match in enumerate(results['abstract_results'][:5], 1):
            output.append(f"\n{i}. [{match['pmid']}] {match.get('title', 'No title')[:60]}...")
            output.append(f"   Hybrid Score: {match['hybrid_score']:.4f}")
            output.append(f"      ├─ BM25:  {match['bm25_score']:.4f}")
            output.append(f"      └─ Dense: {match['dense_score']:.4f}")
            
            meta = match.get('metadata', {})
            output.append(f"   Journal: {meta.get('journal', 'Unknown')}")
            output.append(f"   Year: {meta.get('year', 'N/A')}")
        
        # Full-text results
        if results['searched_fulltext'] and results['fulltext_count'] > 0:
            output.append(f"\n{'='*70}")
            output.append(f"FULL-TEXT RESULTS ({results['fulltext_count']} found)")
            output.append("="*70)
            
            for i, match in enumerate(results['fulltext_results'][:10], 1):
                meta = match['metadata']
                output.append(f"\n{i}. [{meta['pmid']}] Chunk {meta.get('chunk_index', 'N/A')}")
                output.append(f"   Score: {match['score']:.4f}")
                output.append(f"   Section: {meta.get('section', 'Unknown')}")
        
        output.append("\n" + "="*70)
        
        return "\n".join(output)


# ============================================================
# TEST FUNCTION - GÜNCELLENDİ
# ============================================================

def test_hybrid_retrieval_with_expansion():
    """Hybrid retrieval with query expansion'ı test et"""
    
    retriever = HybridMedicrewRetriever()
    
    # Test query'leri (abbreviation ve synonym içeren)
    test_queries = [
        "What are treatments for HF?",  # Abbreviation
        "What is MI diagnosis?",  # Abbreviation  
        "What causes heart attack?",  # Synonym
        "What are ACE inhibitors benefits?",  # Specific term
        "How to treat CHF?",  # Abbreviation
        "ECG findings in AF",  # Abbreviation
        "PCI vs CABG for CAD",  # Multiple abbreviations
    ]
    
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL WITH QUERY EXPANSION TEST")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'='*70}\n")
        results = retriever.retrieve(query)
        print("\n" + retriever.format_results(results))
        print("\n" + "="*70)
        
        # Kullanıcı devam etmek için Enter'a bassın
        try:
            input("\nDevam etmek için Enter'a basın...")
        except:
            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_hybrid_retrieval_with_expansion()