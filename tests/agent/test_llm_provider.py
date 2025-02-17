"""
Tests for LLM providers.
"""

import pytest
import aiohttp
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from deepdroid.agent.llm_provider import (
    LLMProvider,
    OpenRouterProvider,
    OpenAIProvider,
    LLMProviderFactory,
    Message,
    MessageRole
)
import asyncio
from aiohttp import URL

@pytest.fixture
def mock_response():
    """Create a mock API response"""
    return {
        "model": "test-model",
        "choices": [{
            "message": {
                "content": "Test response"
            }
        }],
        "usage": {
            "total_tokens": 10
        }
    }

@pytest.fixture
def mock_session():
    """Create a mock aiohttp session"""
    from unittest.mock import MagicMock
    
    class AsyncContextManagerMock:
        """A custom async context manager for mocking responses"""
        def __init__(self):
            self.status = 200
            self.json = AsyncMock(return_value=None)  # Will be set in individual tests
            self.text = AsyncMock(return_value="Error text")
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Create the response
    response = AsyncContextManagerMock()
    
    # Create the session
    session = MagicMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None
    
    # Create a mock post method that returns our response
    async def mock_post(*args, **kwargs):
        return response
    
    session.post = mock_post
    return session

@pytest.mark.asyncio
async def test_openrouter_provider_success(mock_response, mock_session):
    """Test successful OpenRouter API call"""
    config = {
        'base_url': 'https://test.com',
        'api_key': 'test-key',
        'default_model': 'test-model',
        'timeout': 30
    }
    
    provider = OpenRouterProvider(config)
    messages = [
        Message(role=MessageRole.USER, content="Test message")
    ]
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.post.return_value.json.return_value = mock_response
        response = await provider.generate(messages)
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage == {"total_tokens": 10}
        assert len(response.messages) == 2  # Original + response

@pytest.mark.asyncio
async def test_openrouter_provider_error(mock_session):
    """Test OpenRouter API error handling"""
    config = {
        'base_url': 'https://test.com',
        'api_key': 'test-key',
        'default_model': 'test-model',
        'timeout': 30
    }
    
    provider = OpenRouterProvider(config)
    messages = [
        Message(role=MessageRole.USER, content="Test message")
    ]
    
    # Simulate API error
    mock_session.post.return_value.status = 400
    mock_session.post.return_value.text.return_value = "API Error"
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        with pytest.raises(Exception) as exc_info:
            await provider.generate(messages)
        assert "OpenRouter API error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_openai_provider_success(mock_response, mock_session):
    """Test successful OpenAI API call"""
    config = {
        'base_url': 'https://api.openai.com/v1',
        'api_key': 'test-key',
        'default_model': 'gpt-4',
        'timeout': 30
    }
    
    provider = OpenAIProvider(config)
    messages = [
        Message(role=MessageRole.USER, content="Test message")
    ]
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        mock_session.post.return_value.json.return_value = mock_response
        response = await provider.generate(messages)
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage == {"total_tokens": 10}
        assert len(response.messages) == 2  # Original + response

def test_llm_provider_factory():
    """Test LLM provider factory"""
    config = {
        'openrouter': {
            'base_url': 'https://test.com',
            'api_key': 'test-key',
            'default_model': 'test-model',
            'timeout': 30
        },
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'api_key': 'test-key',
            'default_model': 'gpt-4',
            'timeout': 30
        }
    }
    
    # Test OpenRouter provider creation
    provider = LLMProviderFactory.create('openrouter', config)
    assert isinstance(provider, OpenRouterProvider)
    
    # Test OpenAI provider creation
    provider = LLMProviderFactory.create('openai', config)
    assert isinstance(provider, OpenAIProvider)
    
    # Test invalid provider
    with pytest.raises(ValueError) as exc_info:
        LLMProviderFactory.create('invalid', config)
    assert "Unknown provider" in str(exc_info.value)

def test_provider_missing_api_key():
    """Test provider initialization with missing API key"""
    config = {
        'base_url': 'https://test.com',
        'api_key': '',  # Empty API key
        'default_model': 'test-model',
        'timeout': 30
    }
    
    with pytest.raises(ValueError) as exc_info:
        OpenRouterProvider(config)
    assert "API key not found" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        OpenAIProvider(config)
    assert "API key not found" in str(exc_info.value) 