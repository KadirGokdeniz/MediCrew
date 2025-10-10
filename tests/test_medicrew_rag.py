"""
Test script for MediCrew RAG System
Run this file to test the complete RAG pipeline with sample queries
"""

import sys
import os

# Projenin kök dizinini modül arama yoluna ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Mutlak içe aktarmalar kullan (göreceli değil)
from backend.agents.research_agent.retrival.retrieval_pipeline import HybridMedicrewRetriever
from backend.services.claude_service import ClaudeGenerator, MedicrewRAG

def run_rag_tests():
    """Test Medicrew RAG pipeline with sample queries"""
    
    print("="*70)
    print("MEDICREW RAG SYSTEM TEST")
    print("="*70)
    
    # Initialize components
    retriever = HybridMedicrewRetriever()
    generator = ClaudeGenerator()
    
    # Create RAG pipeline
    rag = MedicrewRAG(retriever, generator)
    
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
    
    # Sonuçları txt dosyasına yazdır
    with open("rag_test_results.txt", "w", encoding="utf-8") as f:
        for i, query in enumerate(test_queries, 1):
            result = rag.query(query)
            formatted_result = rag.format_complete_result(result)
            
            # Dosyaya yaz
            f.write(f"SORU {i}: {query}\n")
            f.write(formatted_result)
            f.write("\n" + "="*70 + "\n\n")
            
            # Aynı zamanda konsola da yazdırmak isterseniz:
            print(f"SORU {i}: {query}")
            print(formatted_result)
            print("\n" + "="*70 + "\n")

    print("Sonuçlar 'rag_test_results.txt' dosyasına kaydedildi!")

if __name__ == "__main__":
    run_rag_tests()