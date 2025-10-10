"""
Test script for SimpleResearcherAgent with Memory
"""

import sys
import os
import time
from datetime import datetime

# Projenin kök dizinini modül arama yoluna ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Modül yollarını ayarlama
from backend.agents.research_agent.researcher_agent import SimpleResearcherAgent

def run_agent_tests():
    """SimpleResearcherAgent'ı çeşitli tıbbi sorgularla test et"""
    
    print("="*70)
    print("SIMPLE RESEARCHER AGENT TEST (Memory Entegrasyonlu)")
    print("="*70)
    
    # Agent'ı başlat (memory kayıt aktif)
    print("\nAgent başlatılıyor...")
    agent = SimpleResearcherAgent(
        temperature=0.3,
        enable_memory_storage=True,
        memory_persistence=True
    )
    print("Agent başlatıldı.\n")
    
    # Benzersiz bir test kullanıcısı
    test_user = f"test_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    agent.set_user_id(test_user)
    
    # Test sorguları - çeşitli tıbbi konular
    test_queries = [
        # guideline / management
        "What are the guideline-recommended first-line treatments for acute decompensated heart failure?",
        "How should ST-elevation myocardial infarction (STEMI) be managed in the first 24 hours according to major cardiology guidelines?",
        "What are the current indications for coronary artery bypass grafting (CABG) versus percutaneous coronary intervention (PCI) in multivessel coronary disease?",
        "Which patients with atrial fibrillation should receive anticoagulation for stroke prevention and how to choose between DOACs and warfarin?",
        "What is the recommended duration of dual antiplatelet therapy after drug-eluting stent placement?",
        
        # diagnostic criteria / interpretation
        "What diagnostic criteria define heart failure with preserved ejection fraction (HFpEF) and which tests confirm it?",
        "How do you interpret troponin dynamics to differentiate acute myocardial infarction from chronic elevation?",
        "Which ECG findings are diagnostic for atrial flutter versus atrial fibrillation, and how to recognize ventricular tachycardia?",
        "What are the echocardiographic features suggestive of hypertrophic cardiomyopathy (HCM)?",
        "What cardiac MRI findings support the diagnosis of myocarditis?",
        
        # therapy comparisons / outcomes
        "What is the comparative effectiveness of ACE inhibitors versus ARBs in chronic heart failure with reduced ejection fraction?",
        "How do outcomes compare between PCI and conservative therapy in stable coronary artery disease?",
        "What is the evidence for using beta-blockers in elderly patients with hypertension and multiple comorbidities?",
        "What are the benefits and harms of PCSK9 inhibitors for secondary prevention in very high-risk patients?",
        "In patients with aortic stenosis, how do transcatheter aortic valve replacement (TAVR) and surgical AVR compare for low-risk older adults?",
        
        # acute care / emergency
        "What is the initial emergency management of suspected acute aortic dissection in the emergency department?",
        "How should cardiogenic shock be stabilized and what are the indications for mechanical circulatory support?",
        "What is the recommended pharmacologic and procedural approach to unstable ventricular tachycardia in the cath lab?",
        "What are best practices for perioperative management of antiplatelet and anticoagulant therapy in patients undergoing non-cardiac surgery?",
        "Which sepsis-like presentations should prompt consideration of infective endocarditis and which immediate investigations are indicated?",
        
        # special populations / scenarios
        "How should heart failure be managed during pregnancy, and what cardiovascular drugs are contraindicated?",
        "What are device therapy indications for cardiac resynchronization therapy (CRT) and implantable cardioverter-defibrillator (ICD) in primary prevention?",
        "How to manage anticoagulation in patients with atrial fibrillation who have recent gastrointestinal bleeding?",
        "What is the evidence-based approach to lipid-lowering therapy in patients with chronic kidney disease?",
        "How to evaluate and manage suspected Takotsubo (stress) cardiomyopathy in the acute setting?",
        
        # guidelines, screening, prevention, and follow-up
        "What are screening recommendations for familial hypercholesterolemia and cascade testing in affected families?",
        "Which lifestyle interventions provide the largest reduction in cardiovascular risk for secondary prevention?",
        "What follow-up testing and monitoring are recommended after discharge for a patient treated for acute myocardial infarction?",
        "How should asymptomatic severe aortic stenosis be monitored and when is elective intervention indicated?",
        "What criteria determine suitability for outpatient management after low-risk syncope with abnormal ECG?"
    ] 
    
    # Test sonuçlarını kaydet
    results_filename = f"agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(results_filename, "w", encoding="utf-8") as f:
        f.write("SIMPLE RESEARCHER AGENT TEST SONUÇLARI (Memory Entegrasyonlu)\n")
        f.write(f"Test Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Kullanıcısı: {test_user}\n")
        f.write("="*70 + "\n\n")
        
        # İlk konuşma
        conversation_id = None
        
        # Her sorgu için agent'ı test et
        for i, query in enumerate(test_queries, 1):
            print(f"\nSORGU {i}/{len(test_queries)}: {query}")
            print("-"*70)
            
            start_time = time.time()
            
            # Agent'ı çalıştır (önceki konuşma ID'sini kullan)
            result = agent.process_query(
                query=query, 
                conversation_id=conversation_id
            )
            
            # İlk konuşma ID'sini kaydet
            if i == 1:
                conversation_id = result["conversation_id"]
                print(f"Konuşma ID: {conversation_id} (takip eden sorgularda kullanılacak)")
            
            # Yanıtı formatla
            formatted = agent.format_response(result)
            
            # İşlem süresini hesapla
            elapsed_time = time.time() - start_time
            
            # Sonucu dosyaya kaydet
            f.write(f"SORGU {i}: {query}\n\n")
            f.write(formatted)
            f.write(f"\n\nİşlem süresi: {elapsed_time:.2f} saniye\n")
            f.write("\n" + "="*70 + "\n\n")
            
            # Konsola özet bilgi yazdır
            print(f"Yanıt alındı. ({elapsed_time:.2f} saniye)")
            print(f"Özet bilgi:")
            print(f"- Abstract sayısı: {result['retrieval_info']['abstract_count']}")
            print(f"- Full-text: {result['retrieval_info']['fulltext_count']}")
            print(f"- Durum: {result['status']}")
            print(f"Sonuç dosyaya kaydedildi.\n")
            
            # Sorguları sırayla çalıştırmak için kısa bir bekleme 
            if i < len(test_queries):
                print("Sonraki sorguya geçiliyor...")
                time.sleep(2)
        
        # Konuşma geçmişini al ve göster
        history = agent.get_conversation_history(conversation_id=conversation_id)
        
        f.write("\nKONUŞMA GEÇMİŞİ:\n")
        f.write(f"Konuşma ID: {conversation_id}\n")
        f.write("-"*70 + "\n")
        
        print("\nKonuşma Geçmişi:")
        print(f"Konuşma ID: {conversation_id}")
        
        if history:
            for entry in history:
                history_line = f"- {entry['timestamp']}: {entry['user_message'][:50]}..."
                f.write(history_line + "\n")
                print(history_line)
        else:
            f.write("Konuşma geçmişi bulunamadı.\n")
            print("Konuşma geçmişi bulunamadı.")
    
    print("\n" + "="*70)
    print(f"Test tamamlandı! Sonuçlar '{results_filename}' dosyasına kaydedildi.")
    print("="*70)

if __name__ == "__main__":
    run_agent_tests()