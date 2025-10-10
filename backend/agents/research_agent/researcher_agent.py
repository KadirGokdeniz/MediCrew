"""
SimpleResearcherAgent - Memory Entegrasyonlu Tıbbi Araştırma Asistanı
Konuşmaları kaydeder ama yanıt oluşturmada kullanmaz
"""

import os
import sys
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Proje kökünü sys.path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Gerekli modülleri import et
from backend.agents.research_agent.retrival.retrieval_pipeline import HybridMedicrewRetriever
from backend.services.claude_service import ClaudeGenerator
from backend.agents.research_agent.memory.memory_manager import MemoryManager  # MemoryManager'ı import et

load_dotenv()

class SimpleResearcherAgent:
    def __init__(
        self, 
        model_name: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4000,
        temperature: float = 0.3,
        enable_memory_storage: bool = True,
        memory_persistence: bool = True,
        memory_collection: str = "medicrew_conversations"
    ):
        """
        Memory entegrasyonlu tıbbi araştırma asistanı
        
        Args:
            model_name: Claude model adı
            max_tokens: Maksimum token sayısı
            temperature: Sıcaklık ayarı
            enable_memory_storage: Konuşmaları bellekte sakla
            memory_persistence: MongoDB'ye kaydet
            memory_collection: MongoDB koleksiyon adı
        """
        print("\n" + "="*70)
        print("🔍 SIMPLE RESEARCHER AGENT (Memory Entegrasyonlu)")
        print("="*70)
        
        # Temel konfigürasyon
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Bellek konfigürasyonu
        self.enable_memory_storage = enable_memory_storage
        self.memory_persistence = memory_persistence
        
        # Oturum bilgisi
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.user_id = "default_user"
        
        # Bileşenleri başlat
        print("\n📚 RAG bileşenleri başlatılıyor...")
        self.retriever = HybridMedicrewRetriever()
        self.generator = ClaudeGenerator()
        
        # Memory Manager'ı başlat (eğer etkinse)
        self.memory_manager = None
        if self.enable_memory_storage:
            print("\n🧠 Memory Manager başlatılıyor...")
            self.memory_manager = MemoryManager(
                collection_name=memory_collection,
                memory_limit=50,
                enable_persistence=memory_persistence,
                enable_semantic_search=False
            )
        
        print("\n✅ SimpleResearcherAgent hazır!")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   Memory Storage: {'Aktif' if enable_memory_storage else 'Devre dışı'}")
        print(f"   MongoDB Persistence: {'Aktif' if memory_persistence else 'Devre dışı'}")
        print(f"   Session ID: {self.session_id}")
        print("="*70 + "\n")
    
    def set_user_id(self, user_id: str):
        """Kullanıcı ID'sini ayarla"""
        self.user_id = user_id
    
    def process_query(self, 
                     query: str, 
                     user_id: Optional[str] = None,
                     conversation_id: Optional[str] = None,
                     force_fulltext: bool = False
                    ) -> Dict:
        """
        Kullanıcı sorgusunu işle ve yanıt döndür
        
        Args:
            query: Kullanıcı sorusu
            user_id: Kullanıcı ID (opsiyonel)
            conversation_id: Konuşma ID (opsiyonel)
            force_fulltext: Full-text aramayı zorla
            
        Returns:
            İşlenmiş yanıt ve metadata
        """
        # Kullanıcı ID'sini belirle
        if user_id:
            self.user_id = user_id
        
        print("\n" + "="*70)
        print(f"📋 YENİ SORGU: {query}")
        print(f"   Kullanıcı: {self.user_id}")
        print(f"   Konuşma ID: {conversation_id or 'Yeni konuşma'}")
        print("="*70)
        
        # 1. RAG Pipeline - Retrieval
        print("\n🔍 Medikal literatür araştırması yapılıyor...")
        retrieval_results = self.retriever.retrieve(
            query, 
            force_fulltext=force_fulltext
        )
        
        # 2. RAG Pipeline - Generation
        print("\n💬 Claude yanıt oluşturuyor...")
        generation_result = self.generator.generate(
            query,
            retrieval_results=retrieval_results,
            temperature=self.temperature,
            max_tokens=self.max_tokens
            # Memory kullanmıyoruz - mevcut ClaudeGenerator bunu desteklemiyor
        )
        
        # 3. Sonucu hazırla
        if generation_result["success"]:
            response_text = generation_result["response"]
            status = "success"
            print("✅ Yanıt başarıyla oluşturuldu.")
        else:
            response_text = f"⚠️ Yanıt üretilirken bir hata oluştu: {generation_result.get('error', 'Bilinmeyen hata')}"
            status = "error"
            print(f"❌ Yanıt oluşturulamadı: {generation_result.get('error', 'Bilinmeyen hata')}")
        
        # 4. Memory'ye kaydet (eğer etkinse)
        if self.enable_memory_storage and self.memory_manager:
            # Metadata oluştur
            metadata = {
                "query_info": {
                    "original_query": query,
                    "force_fulltext": force_fulltext
                },
                "retrieval_info": {
                    "abstract_count": retrieval_results["abstract_count"],
                    "fulltext_count": retrieval_results["fulltext_count"],
                    "routing_decision": retrieval_results["routing_decision"],
                    "expansion_used": retrieval_results.get("expansion_used", False)
                },
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Başarılıysa token bilgilerini ekle
            if status == "success":
                metadata["tokens"] = generation_result["usage"]
            
            # Memory'ye ekle
            new_conversation_id = self.memory_manager.add_memory(
                user_message=query,
                assistant_response=response_text,
                user_id=self.user_id,
                conversation_id=conversation_id,  # Eğer None ise, yeni ID oluşturulur
                metadata=metadata
            )
            
            print(f"🧠 Konuşma memory'ye kaydedildi. ID: {new_conversation_id}")
            conversation_id = new_conversation_id
        
        # 5. Sonucu döndür
        result = {
            "query": query,
            "response": response_text,
            "status": status,
            "conversation_id": conversation_id,
            "retrieval_info": {
                "abstract_count": retrieval_results["abstract_count"],
                "fulltext_count": retrieval_results["fulltext_count"],
                "expansion_used": retrieval_results.get("expansion_used", False),
                "routing_decision": retrieval_results["routing_decision"]
            },
            "generation_info": generation_result if status == "success" else {},
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }
        
        return result
    
    def format_response(self, result: Dict) -> str:
        """
        Sonucu kullanıcıya gösterilecek şekilde formatla
        """
        output = []
        
        # Başlık
        output.append("\n" + "="*70)
        output.append("MEDICREW ARAŞTIRMA SONUCU")
        output.append("="*70)
        
        # Yanıt
        output.append("\n" + result["response"])
        
        # Kaynak bilgisi
        output.append("\n" + "-"*70)
        output.append("KAYNAK BİLGİSİ:")
        output.append(f"• {result['retrieval_info']['abstract_count']} özet")
        output.append(f"• {result['retrieval_info']['fulltext_count']} tam metin bölümü")
        
        if result["retrieval_info"]["expansion_used"]:
            output.append("\n* Tıbbi terim genişletme kullanıldı.")
            
        if result.get("conversation_id"):
            output.append(f"\nKonuşma ID: {result['conversation_id']}")
            
        output.append("\n" + "="*70)
        
        return "\n".join(output)
    
    def get_conversation_history(self, 
                              conversation_id: Optional[str] = None, 
                              user_id: Optional[str] = None,
                              limit: int = 10) -> List:
        """
        Belirli bir konuşmanın veya kullanıcının geçmiş konuşmalarını getir
        
        Not: Bu fonksiyon sadece memory aktifse çalışır
        """
        if not self.enable_memory_storage or not self.memory_manager:
            print("⚠️ Memory devre dışı. Geçmiş konuşmalar alınamıyor.")
            return []
        
        if not user_id:
            user_id = self.user_id
            
        return self.memory_manager.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit
        )
    
    def search_conversation(self, query: str, limit: int = 5) -> List:
        """
        Geçmiş konuşmalarda arama yap
        
        Not: Bu fonksiyon sadece memory aktifse çalışır
        """
        if not self.enable_memory_storage or not self.memory_manager:
            print("⚠️ Memory devre dışı. Konuşmalarda arama yapılamıyor.")
            return []
            
        return self.memory_manager.search_memory(
            query=query,
            user_id=self.user_id,
            search_mode="keyword",
            limit=limit
        )