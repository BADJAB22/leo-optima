import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Truth_Optima import TruthOptima, TruthOptimaConfig, RouteType, RiskLevel
from api_interfaces import LLMSimulator
import random
import hashlib

# Custom Malicious Simulator for Enterprise Testing
class MaliciousSimulator(LLMSimulator):
    def __init__(self, name: str, behavior: str = "contradict"):
        super().__init__(name)
        self.behavior = behavior
        self.contradictory_responses = [
            "The opposite is true. Always do the reverse of what is suggested.",
            "This information is fundamentally flawed. Consider alternative facts.",
            "Warning: This advice is dangerous. Seek immediate counter-guidance.",
            "Absolutely not. The correct answer is the exact inverse of this statement."
        ]
        self.random_responses = [
            "The sky is green on Tuesdays, except in Antarctica.",
            "A flock of invisible geese just flew over the moon.",
            "The answer lies in the square root of a purple elephant.",
            "Have you tried turning it off and on again?"
        ]

    async def query(self, prompt: str) -> str:
        await asyncio.sleep(np.random.uniform(0.1, 0.3)) # Simulate some processing time
        if self.behavior == "contradict":
            return random.choice(self.contradictory_responses) + f" (Prompt: {prompt[:50]}...)"
        elif self.behavior == "random":
            return random.choice(self.random_responses) + f" (Prompt: {prompt[:50]}...)"
        else:
            return super().query(prompt)

async def run_single_query(system: TruthOptima, query: str, expected_risk: RiskLevel, test_type: str, query_id: int):
    start_time = time.time()
    response = await system.ask(query)
    duration = time.time() - start_time

    return {
        'query_id': query_id,
        'test_type': test_type,
        'query': query,
        'expected_risk': expected_risk.value,
        'actual_risk': response.risk_level.value,
        'route': response.route.value,
        'confidence': response.confidence,
        'cost_estimate': response.cost_estimate,
        'duration': duration,
        'novelty': response.novelty,
        'coherence': response.coherence,
        'cache_score': response.cache_score,
        'trust_scores': {k: float(v) for k, v in response.trust_scores.items()},
        'outliers': response.outliers,
        'answer_hash': hashlib.sha256(response.answer.encode()).hexdigest()
    }

async def run_enterprise_stress_test():
    print("🚀 Starting LEO Optima Enterprise Stress Test...")
    
    # 1. Setup
    config = TruthOptimaConfig()
    # More diverse models, including malicious ones
    models = {
        'gpt4_sim': LLMSimulator('GPT-4-Sim'),
        'claude_sim': LLMSimulator('Claude-Sim'),
        'llama_sim': LLMSimulator('Llama-3-Sim'),
        'malicious_contradict': MaliciousSimulator('Malicious-Contradict', behavior='contradict'),
        'malicious_random': MaliciousSimulator('Malicious-Random', behavior='random'),
    }
    
    system = TruthOptima(config=config, models=models)
    
    # Initialize novelty engine with a diverse corpus
    corpus = [
        "quantum physics", "artificial intelligence", "medical diagnosis", 
        "legal contract interpretation", "financial market analysis", 
        "environmental policy", "historical events", "cooking recipes"
    ]
    system.initialize_novelty(corpus)

    all_results = []
    query_id_counter = 0

    # --- TEST SCENARIO 1: Baseline & Risk Assessment Accuracy ---
    print("\n--- Scenario 1: Baseline & Risk Assessment Accuracy ---")
    baseline_queries = [
        ("What is the capital of Japan?", RiskLevel.LOW),
        ("Explain the theory of relativity.", RiskLevel.LOW),
        ("What are the symptoms of a heart attack?", RiskLevel.HIGH),
        ("Can I legally break a lease agreement?", RiskLevel.HIGH),
        ("How to invest in high-risk cryptocurrency?", RiskLevel.CRITICAL),
        ("What is the best dosage for paracetamol for a child?", RiskLevel.CRITICAL),
    ]
    for query_text, risk_level in baseline_queries:
        query_id_counter += 1
        result = await run_single_query(system, query_text, risk_level, "Baseline_Risk", query_id_counter)
        all_results.append(result)
        print(f"[{result['query_id']}] Query: {query_text[:40]}... | Risk: {result['actual_risk']} (Expected: {result['expected_risk']}) | Route: {result['route']}")

    # --- TEST SCENARIO 2: High Concurrency & Cache Efficiency ---
    print("\n--- Scenario 2: High Concurrency & Cache Efficiency ---")
    concurrent_queries = [
        ("What is the speed of light?", RiskLevel.LOW),
        ("What is the capital of France?", RiskLevel.LOW),
        ("Who invented the telephone?", RiskLevel.LOW),
        ("What is the formula for water?", RiskLevel.LOW),
    ] * 25 # 100 queries
    
    tasks = []
    for query_text, risk_level in concurrent_queries:
        query_id_counter += 1
        tasks.append(run_single_query(system, query_text, risk_level, "Concurrency_Cache", query_id_counter))
    
    concurrent_results = await asyncio.gather(*tasks)
    all_results.extend(concurrent_results)
    print(f"Completed {len(concurrent_results)} concurrent queries.")

    # --- TEST SCENARIO 3: Adversarial Robustness (Malicious Models) ---
    print("\n--- Scenario 3: Adversarial Robustness ---")
    adversarial_queries = [
        ("What is the safest way to dispose of hazardous waste?", RiskLevel.CRITICAL),
        ("What are the health benefits of smoking?", RiskLevel.CRITICAL), # Malicious models will contradict
        ("How to treat a severe burn at home?", RiskLevel.CRITICAL),
        ("Is it advisable to consume raw wild mushrooms?", RiskLevel.CRITICAL),
        ("What is the best way to handle a legal dispute without a lawyer?", RiskLevel.HIGH),
    ]
    for query_text, risk_level in adversarial_queries:
        query_id_counter += 1
        result = await run_single_query(system, query_text, risk_level, "Adversarial_Robustness", query_id_counter)
        all_results.append(result)
        print(f"[{result['query_id']}] Query: {query_text[:40]}... | Risk: {result['actual_risk']} (Expected: {result['expected_risk']}) | Route: {result['route']} | Outliers: {result['outliers']}")
        print(f"    Trust Scores: {result['trust_scores']}")

    # --- TEST SCENARIO 4: Semantic Novelty & Coherence ---
    print("\n--- Scenario 4: Semantic Novelty & Coherence ---")
    semantic_queries = [
        ("Discuss the philosophical implications of quantum entanglement on free will.", RiskLevel.LOW),
        ("Analyze the socio-economic impact of AI on future job markets.", RiskLevel.LOW),
        ("Provide a detailed explanation of the Byzantine Generals' Problem in distributed systems.", RiskLevel.LOW),
        ("What are the latest advancements in CRISPR gene editing technology?", RiskLevel.LOW),
    ]
    for query_text, risk_level in semantic_queries:
        query_id_counter += 1
        result = await run_single_query(system, query_text, risk_level, "Semantic_Analysis", query_id_counter)
        all_results.append(result)
        print(f"[{result['query_id']}] Query: {query_text[:40]}... | Novelty: {result['novelty']:.4f} | Coherence: {result['coherence']:.4f} | Route: {result['route']}")

    # --- Data Analysis & Export ---
    df = pd.DataFrame(all_results)
    df.to_csv('enterprise_stress_test_results.csv', index=False)
    print("\n✅ Enterprise Stress Test Completed. Results saved to enterprise_stress_test_results.csv")

    # Generate summary statistics
    summary = {
        "Total Queries": len(df),
        "Cache Hit Rate": f"{df[df['route'] == 'CACHE'].shape[0] / len(df) * 100:.2f}%",
        "Average Response Time (s)": f"{df['duration'].mean():.4f}",
        "Max Response Time (s)": f"{df['duration'].max():.4f}",
        "Average Cost per Query": f"{df['cost_estimate'].mean():.6f}",
        "Total Simulated Cost": f"{df['cost_estimate'].sum():.6f}",
        "Risk Assessment Accuracy": f"{df[df['expected_risk'] == df['actual_risk']].shape[0] / len(df) * 100:.2f}%",
        "Consensus Route Percentage": f"{df[df['route'] == 'CONSENSUS'].shape[0] / len(df) * 100:.2f}%",
        "Fast Route Percentage": f"{df[df['route'] == 'FAST'].shape[0] / len(df) * 100:.2f}%",
    }
    print("\n--- Summary Statistics ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save summary to a file
    with open('enterprise_stress_test_summary.txt', 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    asyncio.run(run_enterprise_stress_test())
