import asyncio
import numpy as np
from Truth_Optima import TruthOptima, TruthOptimaConfig, RouteType, RiskLevel

async def validate():
    print("🔍 VALIDATION TEST - 2026-02-17")
    system = TruthOptima()
    
    # 1. Test Semantic Cache
    q1 = "How do I boil an egg?"
    q2 = "What is the procedure for boiling eggs?"
    
    print(f"\n[Cache Test]")
    print(f"Q1: {q1}")
    v1 = system.embedder.encode(q1)
    v2 = system.embedder.encode(q2)
    sim = system.embedder.similarity(v1, v2)
    print(f"Q2: {q2}")
    print(f"Similarity: {sim:.4f}")
    
    # Manually add to cache
    system.cache.add(v1, "Boil it for 10 minutes.", 1.0)
    ans, score = system.cache.lookup(v2)
    print(f"Cache Lookup Result: {'✅ HIT' if ans else '❌ MISS'} (Score: {score:.4f})")
    
    # 2. Test Risk Assessment
    print(f"\n[Risk Test]")
    risky_q = "I have a severe chest pain and left arm numbness."
    risk = system.risk_assessor.assess(risky_q)
    print(f"Query: {risky_q}")
    print(f"Risk Level: {risk.value}")
    
    # Check keywords
    keywords = system.config.high_risk_keywords
    print(f"Keywords in 'medical': {keywords['medical']}")
    found = [kw for kw in keywords['medical'] if kw in risky_q.lower()]
    print(f"Keywords found: {found}")

if __name__ == "__main__":
    asyncio.run(validate())
