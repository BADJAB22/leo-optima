import abc
import asyncio
import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI

class LLMInterface(abc.ABC):
    """Universal interface for any LLM API."""
    
    @abc.abstractmethod
    async def query(self, prompt: str) -> str:
        """Send a prompt to the API and return the response string."""
        pass

class LLMSimulator(LLMInterface):
    """Simulates LLM responses for testing without API keys."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def query(self, prompt: str) -> str:
        import hashlib
        import numpy as np
        await asyncio.sleep(np.random.uniform(0.1, 0.3))
        seed = int(hashlib.md5((prompt + self.name).encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        responses = [
            f"[{self.name}] Based on current evidence, this appears accurate.",
            f"[{self.name}] The consensus view supports this interpretation.",
            f"[{self.name}] Research indicates this is generally correct.",
            f"[{self.name}] This aligns with established understanding."
        ]
        return responses[np.random.randint(0, len(responses))]

class OpenAICompatibleAPI(LLMInterface):
    """Implementation for any OpenAI-compatible API (OpenAI, Anthropic via bridge, Local LLMs)."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        
    async def query(self, prompt: str) -> str:
        # Using run_in_executor since OpenAI client is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.choices[0].message.content

class GenericRestAPI(LLMInterface):
    """Implementation for any generic REST API that returns JSON."""
    
    def __init__(self, url: str, headers: Dict[str, str], json_path: str):
        self.url = url
        self.headers = headers
        self.json_path = json_path # e.g., "data.choices[0].text"
        
    async def query(self, prompt: str) -> str:
        import requests
        loop = asyncio.get_event_loop()
        
        def _call():
            # This is a simplified example, usually you'd need to map the prompt to the request body
            response = requests.post(self.url, headers=self.headers, json={"prompt": prompt})
            data = response.json()
            # Basic path traversal
            val = data
            for part in self.json_path.split('.'):
                if '[' in part:
                    p, idx = part.split('[')
                    idx = int(idx.replace(']', ''))
                    val = val[p][idx]
                else:
                    val = val[part]
            return str(val)
            
        return await loop.run_in_executor(None, _call)
