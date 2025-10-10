"""
Medicrew - Claude Generation Service
Hybrid retrieval sonuçlarını kullanarak Claude ile yanıt üretir
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # En güncel model

# Generation parameters
MAX_TOKENS = 4096
TEMPERATURE = 0.3  # Medikal domain için düşük (daha deterministik)

# Context limits
MAX_ABSTRACT_CONTEXT = 5  # Kaç abstract'ı context'e dahil edeceğiz
MAX_FULLTEXT_CONTEXT = 10  # Kaç full-text chunk'ı dahil edeceğiz


# ============================================================
# CLAUDE GENERATOR
# ============================================================

class ClaudeGenerator:
    """Claude API ile medikal soru-cevap generation"""
    
    def __init__(self):
        print("🤖 Claude Generator başlatılıyor...")
        
        if not CLAUDE_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY bulunamadı! .env dosyasını kontrol edin.")
        
        self.client = Anthropic(api_key=CLAUDE_API_KEY)
        print(f"✅ Claude hazır (Model: {CLAUDE_MODEL})")
    
    def _build_context_from_abstracts(self, abstract_results: List[Dict]) -> str:
        """Abstract sonuçlarından context oluştur"""
        
        if not abstract_results:
            return "No relevant abstracts found."
        
        context_parts = []
        context_parts.append("# RELEVANT RESEARCH PAPERS\n")
        
        for i, result in enumerate(abstract_results[:MAX_ABSTRACT_CONTEXT], 1):
            pmid = result['pmid']
            title = result.get('title', 'No title')
            text = result.get('text', '')
            
            # Metadata
            meta = result.get('metadata', {})
            journal = meta.get('journal', 'Unknown')
            year = meta.get('year', 'N/A')
            
            # Scores
            hybrid_score = result.get('hybrid_score', 0)
            bm25_score = result.get('bm25_score', 0)
            dense_score = result.get('dense_score', 0)
            
            context_parts.append(f"\n## Paper {i} [PMID: {pmid}]")
            context_parts.append(f"**Title:** {title}")
            context_parts.append(f"**Journal:** {journal} ({year})")
            context_parts.append(f"**Relevance Score:** {hybrid_score:.3f} (BM25: {bm25_score:.3f}, Semantic: {dense_score:.3f})")
            context_parts.append(f"\n**Abstract:**")
            context_parts.append(text[:1500])  # İlk 1500 karakter
            context_parts.append("\n" + "-"*50)
        
        return "\n".join(context_parts)
    
    def _build_context_from_fulltext(self, fulltext_results: List[Dict]) -> str:
        """Full-text chunk'lardan context oluştur"""
        
        if not fulltext_results:
            return ""
        
        context_parts = []
        context_parts.append("\n\n# DETAILED SECTIONS FROM PAPERS\n")
        
        for i, match in enumerate(fulltext_results[:MAX_FULLTEXT_CONTEXT], 1):
            meta = match['metadata']
            pmid = meta['pmid']
            chunk_text = meta.get('text', '')
            section = meta.get('section', 'Unknown')
            score = match['score']
            
            context_parts.append(f"\n## Section {i} [PMID: {pmid}]")
            context_parts.append(f"**Section Type:** {section}")
            context_parts.append(f"**Relevance Score:** {score:.3f}")
            context_parts.append(f"\n**Content:**")
            context_parts.append(chunk_text[:1000])  # İlk 1000 karakter
            context_parts.append("\n" + "-"*50)
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self) -> str:
        """System prompt oluştur"""
        
        return """You are Medicrew, an expert medical research assistant powered by PubMed literature.

Your role is to:
1. Answer medical questions based ONLY on the provided research papers
2. Cite specific PMIDs when making claims
3. Distinguish between high-confidence findings (from abstracts) and detailed evidence (from full-text sections)
4. Acknowledge uncertainty when evidence is limited
5. Use clear, professional medical language

Guidelines:
- Always cite sources with [PMID: XXXXX] format
- If the query was expanded with medical terminology, acknowledge relevant findings
- Prioritize recent, high-quality evidence
- Be concise but comprehensive
- Never make claims without citation
- If evidence is insufficient, say so clearly

Response format:
- Start with a direct answer
- Support with evidence and citations
- End with confidence level and limitations if any"""
    
    def _build_user_prompt(self, query: str, retrieval_results: Dict) -> str:
        """User prompt oluştur (query + context)"""
        
        # Query info
        original_query = retrieval_results['original_query']
        expanded_query = retrieval_results.get('expanded_query', original_query)
        expansion_used = retrieval_results.get('expansion_used', False)
        
        prompt_parts = []
        
        # Query section
        prompt_parts.append(f"# USER QUESTION\n")
        prompt_parts.append(f"**Original Query:** {original_query}")
        
        if expansion_used:
            prompt_parts.append(f"**Expanded Query:** {expanded_query}")
            prompt_parts.append("*(Medical terms were automatically expanded for better retrieval)*")
        
        prompt_parts.append("\n")
        
        # Retrieval info
        prompt_parts.append(f"# RETRIEVAL INFORMATION\n")
        prompt_parts.append(f"- Method: {retrieval_results['retrieval_method']}")
        prompt_parts.append(f"- Abstracts found: {retrieval_results['abstract_count']}")
        prompt_parts.append(f"- Full-text sections found: {retrieval_results['fulltext_count']}")
        prompt_parts.append(f"- Full-text searched: {retrieval_results['searched_fulltext']}")
        prompt_parts.append("\n")
        
        # Context from abstracts
        abstract_context = self._build_context_from_abstracts(
            retrieval_results['abstract_results']
        )
        prompt_parts.append(abstract_context)
        
        # Context from full-text (if available)
        if retrieval_results['searched_fulltext'] and retrieval_results['fulltext_count'] > 0:
            fulltext_context = self._build_context_from_fulltext(
                retrieval_results['fulltext_results']
            )
            prompt_parts.append(fulltext_context)
        
        # Final instruction
        prompt_parts.append("\n\n# INSTRUCTION")
        prompt_parts.append("Based on the research papers provided above, please answer the user's question.")
        prompt_parts.append("Remember to:")
        prompt_parts.append("- Cite all claims with [PMID: XXXXX]")
        prompt_parts.append("- Be concise but thorough")
        prompt_parts.append("- Acknowledge if evidence is limited")
        
        return "\n".join(prompt_parts)
    
    def generate(self, query: str, retrieval_results: Dict, 
                 temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS) -> Dict:
        """
        Claude ile yanıt üret
        
        Args:
            query: Kullanıcı sorusu
            retrieval_results: Hybrid retriever'dan gelen sonuçlar
            temperature: Generation temperature
            max_tokens: Maksimum token sayısı
        
        Returns:
            Dict with response and metadata
        """
        
        print(f"\n🤖 Claude'a gönderiliyor...")
        print(f"   Temperature: {temperature}")
        print(f"   Max tokens: {max_tokens}")
        
        # Check if we have any results
        if retrieval_results['abstract_count'] == 0:
            return {
                'success': False,
                'error': 'no_results',
                'message': 'No relevant papers found for your query.',
                'response': None,
                'timestamp': datetime.now().isoformat()
            }
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, retrieval_results)
        
        try:
            # Call Claude API
            message = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Extract response
            response_text = message.content[0].text
            
            # Usage stats
            usage = {
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
                'total_tokens': message.usage.input_tokens + message.usage.output_tokens
            }
            
            print(f"✅ Yanıt alındı!")
            print(f"   Input tokens: {usage['input_tokens']}")
            print(f"   Output tokens: {usage['output_tokens']}")
            print(f"   Total tokens: {usage['total_tokens']}")
            
            return {
                'success': True,
                'response': response_text,
                'model': CLAUDE_MODEL,
                'temperature': temperature,
                'usage': usage,
                'retrieval_info': {
                    'abstract_count': retrieval_results['abstract_count'],
                    'fulltext_count': retrieval_results['fulltext_count'],
                    'expansion_used': retrieval_results.get('expansion_used', False),
                    'routing_decision': retrieval_results['routing_decision']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Claude API hatası: {e}")
            return {
                'success': False,
                'error': 'api_error',
                'message': str(e),
                'response': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def format_response(self, result: Dict) -> str:
        """Generation sonucunu formatla"""
        
        output = []
        output.append("="*70)
        output.append("MEDICREW - CLAUDE RESPONSE")
        output.append("="*70)
        
        if not result['success']:
            output.append(f"\n❌ Error: {result['error']}")
            output.append(f"Message: {result['message']}")
            return "\n".join(output)
        
        # Response
        output.append(f"\n{result['response']}")
        
        # Metadata
        output.append(f"\n\n{'='*70}")
        output.append("METADATA")
        output.append("="*70)
        output.append(f"Model: {result['model']}")
        output.append(f"Temperature: {result['temperature']}")
        
        # Usage
        usage = result['usage']
        output.append(f"\nTokens:")
        output.append(f"  Input:  {usage['input_tokens']:,}")
        output.append(f"  Output: {usage['output_tokens']:,}")
        output.append(f"  Total:  {usage['total_tokens']:,}")
        
        # Retrieval info
        info = result['retrieval_info']
        output.append(f"\nRetrieval:")
        output.append(f"  Abstracts: {info['abstract_count']}")
        output.append(f"  Full-text sections: {info['fulltext_count']}")
        output.append(f"  Query expansion: {'Yes' if info['expansion_used'] else 'No'}")
        output.append(f"  Routing: {info['routing_decision']}")
        
        output.append(f"\nTimestamp: {result['timestamp']}")
        output.append("="*70)
        
        return "\n".join(output)


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

class MedicrewRAG:
    """Complete RAG pipeline: Retrieval + Generation"""
    
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        print("✅ Medicrew RAG Pipeline hazır!\n")
    
    def query(self, question: str, force_fulltext: bool = False,
              temperature: float = TEMPERATURE) -> Dict:
        """
        End-to-end RAG: Retrieve + Generate
        
        Args:
            question: Kullanıcı sorusu
            force_fulltext: Full-text aramayı zorla
            temperature: Generation temperature
        
        Returns:
            Complete result with retrieval and generation
        """
        
        print("\n" + "="*70)
        print("MEDICREW RAG PIPELINE")
        print("="*70)
        print(f"\nQuestion: {question}\n")
        
        # 1. RETRIEVAL
        print("STEP 1: HYBRID RETRIEVAL WITH QUERY EXPANSION")
        print("-"*70)
        
        retrieval_results = self.retriever.retrieve(
            question, 
            force_fulltext=force_fulltext
        )
        
        # 2. GENERATION
        print("\n" + "-"*70)
        print("STEP 2: CLAUDE GENERATION")
        print("-"*70)
        
        generation_result = self.generator.generate(
            question,
            retrieval_results,
            temperature=temperature
        )
        
        # 3. COMBINE RESULTS
        complete_result = {
            'question': question,
            'retrieval': retrieval_results,
            'generation': generation_result,
            'timestamp': datetime.now().isoformat()
        }
        
        return complete_result
    
    def format_complete_result(self, result: Dict) -> str:
        """Complete result'u formatla"""
        
        output = []
        
        # Header
        output.append("\n" + "="*70)
        output.append("MEDICREW COMPLETE RAG RESULT")
        output.append("="*70)
        output.append(f"\nQuestion: {result['question']}")
        output.append(f"Timestamp: {result['timestamp']}\n")
        
        # Retrieval summary
        ret = result['retrieval']
        output.append("RETRIEVAL SUMMARY:")
        output.append(f"  Original query: {ret['original_query']}")
        if ret.get('expansion_used'):
            output.append(f"  Expanded query: {ret['expanded_query']}")
        output.append(f"  Abstracts found: {ret['abstract_count']}")
        output.append(f"  Full-text sections: {ret['fulltext_count']}")
        output.append(f"  Routing: {ret['routing_decision']}")
        
        # Generation summary
        gen = result['generation']
        output.append(f"\nGENERATION SUMMARY:")
        if gen['success']:
            output.append(f"  Model: {gen['model']}")
            output.append(f"  Tokens: {gen['usage']['total_tokens']:,}")
            output.append(f"  Temperature: {gen['temperature']}")
        else:
            output.append(f"  Status: Failed - {gen['error']}")
        
        # Response
        output.append(f"\n{'='*70}")
        output.append("ANSWER:")
        output.append("="*70 + "\n")
        
        if gen['success']:
            output.append(gen['response'])
        else:
            output.append(f"❌ {gen['message']}")
        
        output.append("\n" + "="*70)
        
        return "\n".join(output)