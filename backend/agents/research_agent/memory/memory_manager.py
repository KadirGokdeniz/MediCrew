"""
Memory Manager - Conversation History Management for AI Assistants
MongoDB persistence, conversation retrieval, and search capabilities
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pymongo
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class MemoryManager:
    def __init__(
        self, 
        collection_name: str = "assistant_conversations",
        memory_limit: int = 20,
        enable_persistence: bool = True,
        embedding_model: Optional[str] = None,
        enable_semantic_search: bool = False,
    ):
        """
        Initialize the memory manager
        
        Args:
            collection_name: MongoDB collection name for storage
            memory_limit: Maximum number of conversations to keep in memory
            enable_persistence: Whether to use MongoDB for persistence
            embedding_model: Model to use for semantic search (if enabled)
            enable_semantic_search: Whether to enable semantic search
        """
        print(f"📚 Memory Manager başlatılıyor...")
        
        # Configuration
        self.memory_limit = memory_limit
        self.enable_persistence = enable_persistence
        self.collection_name = collection_name
        self.enable_semantic_search = enable_semantic_search
        
        # Local memory storage
        self.conversations = []
        
        # MongoDB setup
        if self.enable_persistence:
            self._setup_mongodb(collection_name)
        
        # Semantic search setup
        self.embedding_model = None
        if enable_semantic_search:
            if embedding_model:
                print(f"🧠 Semantic memory için embedding model yükleniyor...")
                try:
                    self.embedding_model = SentenceTransformer(embedding_model)
                    print(f"✅ Embedding model yüklendi: {embedding_model}")
                except Exception as e:
                    print(f"⚠️ Embedding model yüklenemedi: {e}")
                    print("   Semantic search devre dışı bırakıldı.")
                    self.enable_semantic_search = False
            else:
                print("⚠️ Semantic search için model belirtilmedi. Semantic search devre dışı.")
                self.enable_semantic_search = False
        
        print(f"✅ Memory Manager hazır!")
        print(f"   Bellek limiti: {memory_limit} kayıt")
        print(f"   Persistence: {'Aktif' if enable_persistence else 'Devre dışı'}")
        print(f"   Semantic search: {'Aktif' if self.enable_semantic_search else 'Devre dışı'}")
    
    def _setup_mongodb(self, collection_name: str):
        """MongoDB bağlantısını kur"""
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                print("⚠️ MONGO_URI bulunamadı. Persistence devre dışı.")
                self.enable_persistence = False
                self.db = None
                self.collection = None
                return
                
            self.mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client[os.getenv("DB_NAME", "medicrew")]
            self.collection = self.db[collection_name]
            
            # Create indexes for faster retrieval
            self.collection.create_index([("timestamp", pymongo.DESCENDING)])
            self.collection.create_index("user_id")
            self.collection.create_index("conversation_id")
            
            # Check connection
            self.mongo_client.admin.command('ping')
            print(f"✅ MongoDB bağlantısı kuruldu: {collection_name}")
            
        except Exception as e:
            print(f"⚠️ MongoDB bağlantı hatası: {e}")
            self.enable_persistence = False
            self.db = None
            self.collection = None
            print("   Persistence devre dışı bırakıldı.")
    
    def add_memory(
        self,
        user_message: str,
        assistant_response: str,
        user_id: Optional[str] = "anonymous",
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Yeni bir konuşma kaydı ekle
        
        Args:
            user_message: Kullanıcı mesajı
            assistant_response: Asistan yanıtı
            user_id: Kullanıcı kimliği
            conversation_id: Konuşma kimliği (None = yeni konuşma)
            metadata: Ek bilgi
            
        Returns:
            Konuşma ID'si
        """
        timestamp = datetime.now()
        
        # Eğer conversation_id verilmemişse yeni bir tane oluştur
        if conversation_id is None:
            conversation_id = f"conv_{timestamp.strftime('%Y%m%d%H%M%S')}_{user_id}"
        
        # Bellek kaydını oluştur
        memory_entry = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        
        # Semantic embedding (eğer etkinse)
        if self.enable_semantic_search and self.embedding_model:
            try:
                # Kullanıcı mesajı ve asistan yanıtını birleştirerek embed et
                combined_text = f"{user_message} {assistant_response}"
                embedding = self.embedding_model.encode(combined_text)
                memory_entry["embedding"] = embedding.tolist()
            except Exception as e:
                print(f"⚠️ Embedding oluşturma hatası: {e}")
        
        # Yerel belleğe ekle
        self.conversations.append(memory_entry)
        
        # Bellek limitini kontrol et
        if len(self.conversations) > self.memory_limit:
            self.conversations = self.conversations[-self.memory_limit:]
        
        # MongoDB'ye kaydet (eğer etkinse)
        # BU SATIR DEĞİŞTİRİLDİ: Collection objesi doğrudan bool değerlendirmesi yapamaz
        if self.enable_persistence and self.collection is not None:
            try:
                # MongoDB'de _id çakışması olmaması için kaldır
                mongo_entry = memory_entry.copy()
                if "_id" in mongo_entry:
                    del mongo_entry["_id"]
                
                self.collection.insert_one(mongo_entry)
            except Exception as e:
                print(f"⚠️ MongoDB kayıt hatası: {e}")
        
        return conversation_id
    
    def get_conversation_history(
        self, 
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 10,
        include_metadata: bool = False
    ) -> List[Dict]:
        """
        Konuşma geçmişini getir
        
        Args:
            user_id: Belirli bir kullanıcının geçmişini getir
            conversation_id: Belirli bir konuşmayı getir
            limit: Maksimum kayıt sayısı
            include_metadata: Metadata dahil edilsin mi
            
        Returns:
            Konuşma kayıtları listesi
        """
        # Sorgu filtresi oluştur
        query = {}
        if user_id:
            query["user_id"] = user_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        
        # Önce yerel bellekten ara
        results = []
        for entry in self.conversations:
            match = True
            for key, value in query.items():
                if key not in entry or entry[key] != value:
                    match = False
                    break
            
            if match:
                result = entry.copy()
                if not include_metadata:
                    result.pop("metadata", None)
                results.append(result)
        
        # En son olanları önce göster ve limitle
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        local_results = results[:limit]
        
        # Persistence aktifse MongoDB'den de al
        if self.enable_persistence and self.collection is not None:
            try:
                # MongoDB sorgusu
                mongo_results = list(self.collection.find(
                    query, 
                    {"_id": 0} if not include_metadata else None
                ).sort("timestamp", pymongo.DESCENDING).limit(limit))
                
                # Yerel sonuçlarla birleştir
                all_results = mongo_results + local_results
                
                # Tekrarları kaldır (conversation_id ve timestamp'e göre)
                unique_results = {}
                for result in all_results:
                    key = (result["conversation_id"], result["timestamp"].isoformat() if isinstance(result["timestamp"], datetime) else result["timestamp"])
                    unique_results[key] = result
                
                results = list(unique_results.values())
                
                # Zamanlamaya göre sırala ve limitle
                results.sort(key=lambda x: x["timestamp"], reverse=True)
                results = results[:limit]
                
                return results
                
            except Exception as e:
                print(f"⚠️ MongoDB sorgulama hatası: {e}")
                # Eğer hata olursa sadece yerel sonuçları döndür
                return local_results
        
        return local_results
    
    def search_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        search_mode: str = "keyword",
        limit: int = 5
    ) -> List[Dict]:
        """
        Bellek içinde arama yap
        
        Args:
            query: Arama sorgusu
            user_id: Belirli bir kullanıcı ID (opsiyonel)
            search_mode: "keyword" veya "semantic"
            limit: Maksimum sonuç sayısı
            
        Returns:
            Eşleşen konuşma kayıtları listesi
        """
        if search_mode == "semantic" and not self.enable_semantic_search:
            print("⚠️ Semantic search aktif değil, keyword search kullanılıyor.")
            search_mode = "keyword"
        
        # Semantic search (embeddings ile)
        if search_mode == "semantic" and self.embedding_model:
            try:
                # Query embedding oluştur
                query_embedding = self.embedding_model.encode(query)
                
                # Yerel bellekte ara
                results = []
                for entry in self.conversations:
                    if user_id and entry["user_id"] != user_id:
                        continue
                    
                    if "embedding" in entry:
                        # Cosine similarity hesapla
                        similarity = np.dot(query_embedding, entry["embedding"]) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(entry["embedding"])
                        )
                        
                        results.append({
                            "entry": entry,
                            "score": float(similarity)
                        })
                
                # Score'a göre sırala
                results.sort(key=lambda x: x["score"], reverse=True)
                
                # Sadece entry'leri döndür
                return [r["entry"] for r in results[:limit]]
                
            except Exception as e:
                print(f"⚠️ Semantic arama hatası: {e}")
                # Hata durumunda keyword search'e düş
                search_mode = "keyword"
        
        # Keyword search
        if search_mode == "keyword":
            results = []
            query_lower = query.lower()
            
            # Yerel bellekte ara
            for entry in self.conversations:
                if user_id and entry["user_id"] != user_id:
                    continue
                
                # Kullanıcı mesajı veya asistan yanıtında anahtar kelime var mı?
                if (query_lower in entry["user_message"].lower() or 
                    query_lower in entry["assistant_response"].lower()):
                    results.append(entry)
            
            # MongoDB'de ara (eğer etkinse)
            if self.enable_persistence and self.collection is not None:
                try:
                    mongo_query = {
                        "$or": [
                            {"user_message": {"$regex": query_lower, "$options": "i"}},
                            {"assistant_response": {"$regex": query_lower, "$options": "i"}}
                        ]
                    }
                    
                    if user_id:
                        mongo_query["user_id"] = user_id
                    
                    mongo_results = list(self.collection.find(
                        mongo_query, {"_id": 0}
                    ).sort("timestamp", pymongo.DESCENDING).limit(limit))
                    
                    # Yerel sonuçlarla birleştir
                    all_results = mongo_results + results
                    
                    # Tekrarları kaldır
                    unique_results = {}
                    for result in all_results:
                        key = (result["conversation_id"], str(result["timestamp"]))
                        unique_results[key] = result
                    
                    results = list(unique_results.values())
                    
                except Exception as e:
                    print(f"⚠️ MongoDB arama hatası: {e}")
            
            # En son konuşmaları önce göster
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results[:limit]
        
        return []
    
    def format_for_context(
        self,
        entries: List[Dict],
        format_type: str = "text",
        max_length: int = 1500
    ) -> str:
        """
        Bellek girdilerini Claude'un context'i için formatla
        
        Args:
            entries: Bellek girdileri listesi
            format_type: "text" veya "markdown"
            max_length: Maksimum karakter uzunluğu
            
        Returns:
            Formatlı bellek metni
        """
        if not entries:
            return ""
        
        # Bellek girdilerini zamana göre sırala (en eskiler önce)
        entries.sort(key=lambda x: x["timestamp"])
        
        # Format seçimi
        if format_type == "markdown":
            memory_text = "\n\n## GEÇMİŞ KONUŞMALAR\n\n"
            
            for i, entry in enumerate(entries):
                memory_text += f"### Konuşma {i+1}\n"
                memory_text += f"**Kullanıcı:** {entry['user_message']}\n\n"
                
                # Uzun yanıtları kısalt
                response = entry['assistant_response']
                if len(response) > 300:
                    response = response[:250] + "... [kısaltıldı]"
                    
                memory_text += f"**Asistan:** {response}\n\n"
                memory_text += "---\n"
        else:
            memory_text = "\n\nGEÇMİŞ KONUŞMALAR:\n\n"
            
            for i, entry in enumerate(entries):
                memory_text += f"Konuşma {i+1}:\n"
                memory_text += f"Kullanıcı: {entry['user_message']}\n"
                
                # Uzun yanıtları kısalt
                response = entry['assistant_response']
                if len(response) > 300:
                    response = response[:250] + "... [kısaltıldı]"
                    
                memory_text += f"Asistan: {response}\n\n"
        
        # Belirli bir uzunluğu aşmasını engelle
        if len(memory_text) > max_length:
            # Son konuşmayı koru, gerisini kısalt
            half_length = max_length // 2
            memory_text = memory_text[:half_length] + "\n[...]\n" + memory_text[-half_length:]
        
        return memory_text
    
    def clear_memory(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        older_than: Optional[int] = None
    ) -> int:
        """
        Belleği temizle
        
        Args:
            user_id: Belirli bir kullanıcının verilerini temizle
            conversation_id: Belirli bir konuşmayı temizle
            older_than: Belirtilen günden daha eski kayıtları temizle
            
        Returns:
            Temizlenen kayıt sayısı
        """
        # Yerel bellekte temizlik
        original_count = len(self.conversations)
        filtered_conversations = []
        
        for entry in self.conversations:
            should_keep = True
            
            if user_id and entry["user_id"] == user_id:
                should_keep = False
            
            if conversation_id and entry["conversation_id"] == conversation_id:
                should_keep = False
            
            if older_than:
                cutoff_date = datetime.now() - timedelta(days=older_than)
                entry_date = entry["timestamp"] if isinstance(entry["timestamp"], datetime) else datetime.fromisoformat(entry["timestamp"])
                if entry_date < cutoff_date:
                    should_keep = False
            
            if should_keep:
                filtered_conversations.append(entry)
        
        self.conversations = filtered_conversations
        local_removed = original_count - len(self.conversations)
        
        # MongoDB'de temizlik (eğer etkinse)
        mongo_removed = 0
        if self.enable_persistence and self.collection is not None:
            try:
                query = {}
                
                if user_id:
                    query["user_id"] = user_id
                
                if conversation_id:
                    query["conversation_id"] = conversation_id
                
                if older_than:
                    cutoff_date = datetime.now() - timedelta(days=older_than)
                    query["timestamp"] = {"$lt": cutoff_date}
                
                if query:
                    result = self.collection.delete_many(query)
                    mongo_removed = result.deleted_count
                    
            except Exception as e:
                print(f"⚠️ MongoDB temizleme hatası: {e}")
        
        total_removed = local_removed + mongo_removed
        print(f"🧹 Bellek temizlendi: {total_removed} kayıt silindi")
        
        return total_removed
    
    def get_statistics(self, user_id: Optional[str] = None) -> Dict:
        """
        Bellek istatistiklerini getir
        
        Args:
            user_id: Belirli bir kullanıcı için istatistikler (opsiyonel)
            
        Returns:
            İstatistikler sözlüğü
        """
        stats = {
            "local_count": 0,
            "persistent_count": 0,
            "user_count": 0,
            "conversation_count": 0,
            "oldest_entry": None,
            "newest_entry": None,
            "average_message_length": 0,
            "average_response_length": 0
        }
        
        # Yerel bellek istatistikleri
        user_ids = set()
        conversation_ids = set()
        total_message_length = 0
        total_response_length = 0
        oldest = None
        newest = None
        
        filtered_conversations = self.conversations
        if user_id:
            filtered_conversations = [c for c in self.conversations if c["user_id"] == user_id]
        
        for entry in filtered_conversations:
            user_ids.add(entry["user_id"])
            conversation_ids.add(entry["conversation_id"])
            
            total_message_length += len(entry["user_message"])
            total_response_length += len(entry["assistant_response"])
            
            entry_date = entry["timestamp"] if isinstance(entry["timestamp"], datetime) else datetime.fromisoformat(entry["timestamp"])
            
            if oldest is None or entry_date < oldest:
                oldest = entry_date
            
            if newest is None or entry_date > newest:
                newest = entry_date
        
        stats["local_count"] = len(filtered_conversations)
        stats["user_count"] = len(user_ids)
        stats["conversation_count"] = len(conversation_ids)
        
        if filtered_conversations:
            stats["average_message_length"] = total_message_length / len(filtered_conversations)
            stats["average_response_length"] = total_response_length / len(filtered_conversations)
        
        if oldest:
            stats["oldest_entry"] = oldest.isoformat()
        
        if newest:
            stats["newest_entry"] = newest.isoformat()
        
        # MongoDB istatistikleri (eğer etkinse)
        if self.enable_persistence and self.collection is not None:
            try:
                query = {}
                if user_id:
                    query["user_id"] = user_id
                
                stats["persistent_count"] = self.collection.count_documents(query)
                
                if not query and not filtered_conversations:
                    # Global istatistikleri alıyoruz ve yerel bellek boşsa
                    stats["user_count"] = len(self.collection.distinct("user_id"))
                    stats["conversation_count"] = len(self.collection.distinct("conversation_id"))
                    
                    oldest_entry = self.collection.find_one(sort=[("timestamp", pymongo.ASCENDING)])
                    if oldest_entry:
                        stats["oldest_entry"] = oldest_entry["timestamp"].isoformat()
                    
                    newest_entry = self.collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
                    if newest_entry:
                        stats["newest_entry"] = newest_entry["timestamp"].isoformat()
                
            except Exception as e:
                print(f"⚠️ MongoDB istatistik hatası: {e}")
        
        return stats

# Örnek kullanım
if __name__ == "__main__":
    # Test
    memory = MemoryManager(
        collection_name="test_conversations",
        memory_limit=10,
        enable_persistence=True,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        enable_semantic_search=True
    )
    
    # Örnek konuşma ekle
    conversation_id = memory.add_memory(
        user_message="Kalp krizinin belirtileri nelerdir?",
        assistant_response="Kalp krizinin belirtileri arasında göğüs ağrısı, nefes darlığı, sol kola, çeneye veya sırta yayılan ağrı sayılabilir.",
        user_id="test_user",
        metadata={"topic": "cardiology", "importance": "high"}
    )
    
    # Aynı konuşmaya ek yap
    memory.add_memory(
        user_message="Teşekkür ederim. Peki kalp krizi geçiren birine nasıl yardım edebilirim?",
        assistant_response="Kalp krizi geçirdiğinden şüphelendiğiniz birini gördüğünüzde, hemen 112'yi arayın ve kişiyi rahat bir pozisyonda tutun.",
        user_id="test_user",
        conversation_id=conversation_id
    )
    
    # Konuşma geçmişini göster
    history = memory.get_conversation_history(user_id="test_user", limit=5)
    print("\nKonuşma Geçmişi:")
    for entry in history:
        print(f"- {entry['timestamp']}: {entry['user_message'][:30]}...")
    
    # Arama yap
    search_results = memory.search_memory("kalp krizi belirtileri")
    print("\nArama Sonuçları:")
    for entry in search_results:
        print(f"- {entry['user_message'][:30]}...")
    
    # Context formatı
    context = memory.format_for_context(search_results, format_type="markdown")
    print("\nContext Formatı:")
    print(context[:300] + "...")
    
    # İstatistikler
    stats = memory.get_statistics()
    print("\nBellek İstatistikleri:")
    print(json.dumps(stats, indent=2))