"""
Smart Risk Assessment for LEO Optima
=====================================
Uses LLM-based contextual analysis to accurately assess query risk levels.
Model Agnostic: Works with any LLM provider (OpenAI, Anthropic, Local, etc.)
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SmartRiskAssessor:
    """
    Intelligent risk assessment using LLM with caching.
    Works with any LLM interface provided to the system.
    """
    
    RISK_ASSESSMENT_PROMPT = """You are a risk assessment AI. Classify the risk level of the following user query.
    
    Risk Levels:
    - CRITICAL: Immediate emergencies, life-threatening, illegal acts, or dangerous medical advice.
    - HIGH: Medical, legal, financial decisions, or professional expertise needed.
    - LOW: General info, facts, or safe how-to guides.

    Respond with ONLY one word: CRITICAL, HIGH, or LOW.

    Query: {query}
    
    Risk Level:"""
    
    def __init__(
        self,
        config: Any = None,
        model_interface: Any = None,
        cache_dir: str = "leo_storage/risk_cache"
    ):
        self.config = config
        self.model_interface = model_interface
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # Fallback keyword lists (from original implementation)
        self.fallback_keywords = {
            'medical': ['medical', 'health', 'disease', 'drug', 'medicine', 'treatment', 'chest pain', 'heart'],
            'legal': ['legal', 'law', 'court', 'sue', 'contract', 'rights'],
            'financial': ['invest', 'stock', 'crypto', 'loan', 'tax', 'money'],
            'safety': ['safe', 'danger', 'risk', 'emergency', 'hazard', 'kill', 'suicide']
        }

    def _get_cache_path(self, query: str) -> str:
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    async def assess(self, prompt: str, use_cache: bool = True) -> RiskLevel:
        if use_cache:
            cache_path = self._get_cache_path(prompt)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                        return RiskLevel(data['risk_level'])
                except:
                    pass

        # Try LLM Assessment if interface is available
        if self.model_interface:
            try:
                formatted_prompt = self.RISK_ASSESSMENT_PROMPT.format(query=prompt)
                # Call the model interface (assuming it has a call or generate method)
                # LEO Optima's api_interfaces typically use 'generate' or 'ask'
                if hasattr(self.model_interface, 'query'):
                    result = await self.model_interface.query(formatted_prompt)
                elif hasattr(self.model_interface, 'generate'):
                    result = await self.model_interface.generate(formatted_prompt)
                else:
                    # Fallback to direct call if it's a simple function
                    result = await self.model_interface(formatted_prompt)
                
                result = result.strip().upper()
                
                risk_level = RiskLevel.LOW
                if "CRITICAL" in result:
                    risk_level = RiskLevel.CRITICAL
                elif "HIGH" in result:
                    risk_level = RiskLevel.HIGH
                
                if use_cache:
                    with open(self._get_cache_path(prompt), 'w') as f:
                        json.dump({
                            'query': prompt,
                            'risk_level': risk_level.value,
                            'timestamp': datetime.now().isoformat()
                        }, f)
                return risk_level
            except Exception as e:
                print(f"⚠️ Smart Risk LLM assessment failed: {e}. Using keywords.")

        # Fallback to Keyword Matching
        return self._assess_with_keywords(prompt)

    def _assess_with_keywords(self, prompt: str) -> RiskLevel:
        prompt_lower = prompt.lower()
        hits = 0
        for category, keywords in self.fallback_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                hits += 1
        
        if hits >= 2: return RiskLevel.CRITICAL
        if hits == 1: return RiskLevel.HIGH
        return RiskLevel.LOW
