import pytest
import asyncio
from Truth_Optima import TruthOptima, TruthOptimaConfig, RouteType, RiskLevel
from api_interfaces import LLMSimulator

@pytest.fixture
def config():
    return TruthOptimaConfig()

@pytest.fixture
def simulators():
    return {
        'gpt4': LLMSimulator('GPT-4-Sim'),
        'claude': LLMSimulator('Claude-Sim'),
    }

@pytest.mark.asyncio
async def test_truth_optima_initialization(config, simulators):
    system = TruthOptima(config=config, models=simulators)
    assert system is not None
    assert isinstance(system, TruthOptima)
    assert len(system.models) == 2

@pytest.mark.asyncio
async def test_truth_optima_ask_method(config, simulators):
    system = TruthOptima(config=config, models=simulators)
    query = "What is the capital of France?"
    response = await system.ask(query)
    
    assert response is not None
    assert isinstance(response.answer, str)
    assert response.route in [RouteType.CACHE, RouteType.FAST, RouteType.CONSENSUS]
    assert response.risk_level in [RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert response.confidence >= 0.0 and response.confidence <= 1.0

@pytest.mark.asyncio
async def test_truth_optima_cache_hit(config, simulators):
    system = TruthOptima(config=config, models=simulators)
    query = "Tell me about quantum entanglement."
    
    # First query should not be a cache hit
    response1 = await system.ask(query)
    assert response1.route != RouteType.CACHE
    
    # Second query with same prompt should be a cache hit
    response2 = await system.ask(query)
    assert response2.route == RouteType.CACHE
    assert response2.answer == response1.answer
