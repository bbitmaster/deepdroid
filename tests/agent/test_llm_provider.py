"""
Tests for LLM providers.
"""

import pytest
from deepdroid.agent.llm_provider import (
    LLMProvider,
    OpenRouterProvider,
    OpenAIProvider,
    LLMProviderFactory,
    Message,
    MessageRole,
    format_chat_messages,
    parse_chat_response
)

def test_format_chat_messages():
    """Test message formatting for chat APIs"""
    messages = [
        Message(role=MessageRole.SYSTEM, content="System message"),
        Message(role=MessageRole.USER, content="User message"),
        Message(role=MessageRole.ASSISTANT, content="Assistant message"),
        Message(role=MessageRole.TOOL, content="Tool output", name="tool_name")
    ]
    
    formatted = format_chat_messages(messages)
    
    assert len(formatted) == 4
    assert formatted[0] == {
        'role': 'system',
        'content': 'System message'
    }
    assert formatted[1] == {
        'role': 'user',
        'content': 'User message'
    }
    assert formatted[2] == {
        'role': 'assistant',
        'content': 'Assistant message'
    }
    assert formatted[3] == {
        'role': 'tool',
        'content': 'Tool output',
        'name': 'tool_name'
    }

def test_parse_chat_response():
    """Test parsing chat API responses"""
    original_messages = [
        Message(role=MessageRole.USER, content="Test message")
    ]
    
    response_data = {
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
    
    result = parse_chat_response(response_data, original_messages)
    
    assert result.content == "Test response"
    assert result.model == "test-model"
    assert result.usage == {"total_tokens": 10}
    assert len(result.messages) == 2
    assert result.messages[0].role == MessageRole.USER
    assert result.messages[1].role == MessageRole.ASSISTANT
    assert result.messages[1].content == "Test response"

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