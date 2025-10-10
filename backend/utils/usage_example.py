"""
Mevcut Retrieval Pipeline Performance Analysis
Retrieval sonuçlarını analiz et ve iyileştirme önerileri sun
"""

from retrieval_pipeline import HybridMedicrewRetriever
import numpy as np

def analyze_current_performance():
    """Mevcut retrieval pipeline performansını analiz et"""
    
    retriever = HybridMedicrewRetriever()
    
    comprehensive_queries = [
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

    
    print("=" * 80)
    print("CURRENT RETRIEVAL PIPELINE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    performance_stats = []
    
    for query in comprehensive_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        result = retriever.retrieve(query)
        
        stats = {
            'query': query,
            'top_score': result['abstract_results'][0]['hybrid_score'] if result['abstract_results'] else 0,
            'abstract_count': result['abstract_count'],
            'routing': result['routing_decision'],
            'fulltext_searched': result['searched_fulltext'],
            'expansion_used': result['expansion_used']
        }
        
        performance_stats.append(stats)
        
        print(f"Top Score: {stats['top_score']:.4f}")
        print(f"Abstracts Found: {stats['abstract_count']}")
        print(f"Routing: {stats['routing']}")
        print(f"Expansion Used: {stats['expansion_used']}")
        
        # Performans değerlendirmesi
        if stats['top_score'] >= 0.7:
            print("✅ MÜKEMMEL PERFORMANS")
        elif stats['top_score'] >= 0.6:
            print("✅ İYİ PERFORMANS") 
        elif stats['top_score'] >= 0.5:
            print("⚠️  ORTA PERFORMANS")
        else:
            print("❌ DÜŞÜK PERFORMANS")
    
    # Genel istatistikler
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    scores = [s['top_score'] for s in performance_stats]
    print(f"Total Queries: {len(performance_stats)}")
    print(f"Average Score: {np.mean(scores):.4f}")
    print(f"Median Score:  {np.median(scores):.4f}")
    print(f"Min Score:     {min(scores):.4f}")
    print(f"Max Score:     {max(scores):.4f}")
    
    # Score distribution
    excellent = len([s for s in scores if s >= 0.7])
    good = len([s for s in scores if 0.6 <= s < 0.7])
    medium = len([s for s in scores if 0.5 <= s < 0.6])
    poor = len([s for s in scores if s < 0.5])
    
    print(f"\nPerformance Distribution:")
    print(f"Excellent (≥0.7): {excellent}/{len(scores)} ({excellent/len(scores)*100:.1f}%)")
    print(f"Good (0.6-0.7):   {good}/{len(scores)} ({good/len(scores)*100:.1f}%)")
    print(f"Medium (0.5-0.6): {medium}/{len(scores)} ({medium/len(scores)*100:.1f}%)")
    print(f"Poor (<0.5):      {poor}/{len(scores)} ({poor/len(scores)*100:.1f}%)")
    
    # İyileştirme önerileri
    print("\n" + "=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    poor_queries = [s for s in performance_stats if s['top_score'] < 0.5]
    if poor_queries:
        print("🚨 Düşük Performanslı Query'ler:")
        for pq in poor_queries:
            print(f"  - '{pq['query']}' (Score: {pq['top_score']:.4f})")
        print("\n✅ Öneriler:")
        print("  1. Dictionary'yi genişlet")
        print("  2. BM25 ağırlığını artır")
        print("  3. Threshold'u düşür")
        print("  4. Query expansion stratejisini iyileştir")
    else:
        print("🎉 Tüm query'ler makul performansta!")

if __name__ == "__main__":
    analyze_current_performance()
