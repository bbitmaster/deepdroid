"""
Tests for the Agent class.

This module contains comprehensive tests for the Agent class, including:
- Basic functionality tests
- Conversation management tests
- Error handling tests
- Configuration tests
- Tool output handling tests
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
import os
import tempfile
import yaml
from unittest.mock import Mock, patch
from deepdroid.agent.agent import Agent
from deepdroid.agent.llm_provider import (
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole
)
from deepdroid.parser import Parser
from deepdroid.agent.llm_provider import OpenRouterProvider, OpenAIProvider

# Mock classes for testing
class MockParser(Parser):
    """
    Mock parser for testing that tracks parse calls and cleanup status.
    
    Attributes:
        memory_file: Path to the memory file
        parse_calls: List of responses passed to parse()
        cleanup_called: Whether cleanup() was called
    """
    def __init__(self, memory_file: str = "test_memory.txt"):
        super().__init__()
        self.memory_file = memory_file
        self.parse_calls = []
        self.cleanup_called = False
    
    def parse(self, response: str) -> str:
        self.parse_calls.append(response)
        return f"Parsed: {response}"
    
    def cleanup(self) -> None:
        self.cleanup_called = True

class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing that returns predefined responses.
    
    Attributes:
        responses: List of predefined response dictionaries
        response_index: Current index in the responses list
        calls: List of calls made to generate()
    """
    def __init__(self, responses: List[Dict[str, Any]] = None):
        self.responses = responses or [{
            "content": "Default mock response",
            "model": "mock-model",
            "usage": {"total_tokens": 10}
        }]
        self.response_index = 0
        self.calls = []
    
    async def generate(self,
                      messages: List[Message],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        response = self.responses[self.response_index]
        self.response_index = (self.response_index + 1) % len(self.responses)
        
        updated_messages = messages + [Message(
            role=MessageRole.ASSISTANT,
            content=response["content"]
        )]
        
        return LLMResponse(
            content=response["content"],
            model=response["model"],
            usage=response["usage"],
            raw_response=response,
            messages=updated_messages
        )

# Configuration fixtures
@pytest.fixture
def base_config():
    """Base configuration dictionary used by other config fixtures"""
    return {
        "llm": {
            "provider": "mock",
            "mock": {}
        },
        "agent": {
            "max_retries": 2,
            "retry_delay": 0.1,
            "memory_file": "test_memory.txt",
            "system_prompt_dir": "test_prompts",
            "log_level": "DEBUG",
            "conversation": {
                "max_turns": 3,
                "max_chars": 1000,
                "include_tool_outputs": True,
                "summarize_on_reset": True
            }
        }
    }

@pytest.fixture
def mock_config_with_tools(base_config):
    """Config fixture with tool outputs enabled"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(base_config, f)
        config_path = f.name
    yield config_path
    os.unlink(config_path)

@pytest.fixture
def mock_config_no_tools(base_config):
    """Config fixture with tool outputs disabled"""
    config = base_config.copy()
    config['agent']['conversation']['include_tool_outputs'] = False
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    yield config_path
    os.unlink(config_path)

@pytest.fixture
def mock_config(mock_config_with_tools):
    """Alias for mock_config_with_tools for backward compatibility"""
    return mock_config_with_tools

# System prompt and LLM provider fixtures
@pytest.fixture
def mock_system_prompt(mock_config, monkeypatch):
    """Creates a temporary system prompt file and patches the config manager"""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_path = os.path.join(temp_dir, "test_prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write("Test system prompt for testing purposes")
        
        def mock_get_system_prompt(self, prompt_name: str) -> str:
            full_path = os.path.join(temp_dir, f"{prompt_name}.txt")
            with open(full_path, 'r') as f:
                return f.read()
        
        monkeypatch.setattr(
            'deepdroid.agent.config.ConfigManager.get_system_prompt',
            mock_get_system_prompt
        )
        
        yield prompt_path

@pytest.fixture
def mock_llm_provider():
    """Creates a mock LLM provider with predefined test responses"""
    return MockLLMProvider([
        {
            "content": "Test response 1",
            "model": "mock-model",
            "usage": {"total_tokens": 10}
        },
        {
            "content": "Test response 2",
            "model": "mock-model",
            "usage": {"total_tokens": 15}
        },
        {
            "content": "Conversation summary",
            "model": "mock-model",
            "usage": {"total_tokens": 5}
        }
    ])

@pytest.fixture
def mock_llm_factory(mock_llm_provider, monkeypatch):
    """Patches the LLMProviderFactory to use our mock provider"""
    providers = {
        'openrouter': OpenRouterProvider,
        'openai': OpenAIProvider,
        'mock': lambda config: mock_llm_provider
    }
    monkeypatch.setattr(
        'deepdroid.agent.llm_provider.LLMProviderFactory.create',
        lambda provider, config: providers[provider](config)
    )
    return providers

# Basic functionality tests
@pytest.mark.asyncio
async def test_agent_initialization(mock_config_no_tools, mock_system_prompt, mock_llm_factory):
    """Test agent initialization with config"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    assert agent.max_retries == 2
    assert agent.retry_delay == 0.1
    assert isinstance(agent.parser, MockParser)
    assert len(agent.messages) == 0
    assert agent.turn_count == 0
    assert isinstance(agent.llm, MockLLMProvider)

@pytest.mark.asyncio
async def test_process_message_basic(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test basic message processing"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt"
    )
    
    # Check message history
    assert len(agent.messages) == 3  # system prompt + user message + assistant response
    assert agent.messages[0].role == MessageRole.SYSTEM
    assert agent.messages[1].role == MessageRole.USER
    assert agent.messages[2].role == MessageRole.ASSISTANT
    
    # Check LLM was called correctly
    assert len(agent.llm.calls) == 1
    assert agent.llm.calls[0]["messages"][1].content == "Test message"
    
    # Check parser was called
    assert len(agent.parser.parse_calls) == 1
    assert agent.parser.parse_calls[0] == "Test response 1"
    
    # Check result
    assert result == "Parsed: Test response 1"

@pytest.mark.asyncio
async def test_conversation_reset_on_max_turns(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test conversation reset when max turns is reached"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    
    # Process messages up to max_turns
    for i in range(3):  # max_turns is 3
        result = await agent.process_message(
            message=f"Message {i}",
            system_prompt_name="test_prompt"
        )
    
    # This should trigger a reset
    result = await agent.process_message(
        message="Message after reset",
        system_prompt_name="test_prompt"
    )
    
    # Check that conversation was reset
    assert agent.turn_count == 1
    assert len(agent.messages) == 3  # system prompt + user message + assistant response
    assert "Previous conversation summary" in agent.messages[0].content

@pytest.mark.asyncio
async def test_conversation_reset_on_max_chars(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test conversation reset when max characters is reached"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    
    # Send a large message that exceeds max_chars
    large_message = "x" * 1001  # max_chars is 1000
    result = await agent.process_message(
        message=large_message,
        system_prompt_name="test_prompt"
    )
    
    # Check that conversation was reset
    assert agent.turn_count == 1
    assert len(agent.messages) == 3  # system prompt + user message + assistant response
    assert agent.messages[0].role == MessageRole.SYSTEM
    assert agent.messages[1].role == MessageRole.USER
    assert agent.messages[2].role == MessageRole.ASSISTANT

@pytest.mark.asyncio
async def test_tool_output_inclusion(mock_config_with_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test that tool outputs are included in conversation history"""
    agent = Agent(MockParser, config_path=mock_config_with_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt"
    )
    
    # Check that tool output was added to history
    assert len(agent.messages) == 4  # system + user + assistant + tool
    assert agent.messages[-1].role == MessageRole.TOOL
    assert agent.messages[-1].name == "command_output"
    assert agent.messages[-1].content == "Parsed: Test response 1"

@pytest.mark.asyncio
async def test_llm_retry_on_failure(mock_config_no_tools, mock_system_prompt, mock_llm_factory):
    """Test that agent retries on LLM failure"""
    failing_provider = MockLLMProvider()
    failing_provider.generate = Mock(side_effect=[
        Exception("Test error"),
        MockLLMProvider().generate([], 0.7)  # Succeed on second try
    ])
    
    # Override the mock provider in the factory for this test
    mock_llm_factory['mock'] = lambda config: failing_provider
    
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt"
    )
    
    # Check that generate was called twice (one failure, one success)
    assert failing_provider.generate.call_count == 2
    
    # Verify we got a successful result
    assert result == "Parsed: Default mock response"

@pytest.mark.asyncio
async def test_cleanup(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test cleanup is called when using context manager"""
    async with Agent(MockParser, config_path=mock_config_no_tools) as agent:
        await agent.process_message(
            message="Test message",
            system_prompt_name="test_prompt"
        )
    
    assert agent.parser.cleanup_called

@pytest.mark.asyncio
async def test_custom_temperature(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test custom temperature setting"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt",
        temperature=0.5
    )
    
    assert agent.llm.calls[0]["temperature"] == 0.5

@pytest.mark.asyncio
async def test_custom_max_tokens(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test custom max_tokens setting"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt",
        max_tokens=100
    )
    
    assert agent.llm.calls[0]["max_tokens"] == 100

@pytest.mark.asyncio
async def test_missing_system_prompt(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test handling of missing system prompt file"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    with pytest.raises(Exception) as exc_info:
        await agent.process_message(
            message="Test message",
            system_prompt_name="nonexistent_prompt"
        )
    error_msg = str(exc_info.value)
    # Check for either the wrapped error message or the raw FileNotFoundError
    assert any(msg in error_msg for msg in [
        "Error loading system prompt",
        "No such file or directory",
        "nonexistent_prompt.txt"
    ])

@pytest.mark.asyncio
async def test_empty_message(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test handling of empty message"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="",
        system_prompt_name="test_prompt"
    )
    assert result == "Parsed: Test response 1"

@pytest.mark.asyncio
async def test_consecutive_resets(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test handling of consecutive conversation resets"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    
    # First reset
    large_message = "x" * 1001
    result1 = await agent.process_message(
        message=large_message,
        system_prompt_name="test_prompt"
    )
    
    # Immediate second reset
    result2 = await agent.process_message(
        message=large_message,
        system_prompt_name="test_prompt"
    )
    
    # Both should work without errors
    assert agent.turn_count == 1
    assert len(agent.messages) == 3  # system + user + assistant

@pytest.mark.asyncio
async def test_max_retries_exhausted(mock_config_no_tools, mock_system_prompt, mock_llm_factory):
    """Test that agent raises error when max retries are exhausted"""
    failing_provider = MockLLMProvider()
    failing_provider.generate = Mock(side_effect=Exception("Persistent error"))
    
    # Override the mock provider in the factory for this test
    mock_llm_factory['mock'] = lambda config: failing_provider
    
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    with pytest.raises(Exception) as exc_info:
        await agent.process_message(
            message="Test message",
            system_prompt_name="test_prompt"
        )
    assert "Persistent error" in str(exc_info.value)
    assert failing_provider.generate.call_count == 2  # max_retries is 2

@pytest.mark.asyncio
async def test_conversation_state_persistence(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test that conversation state persists between messages"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    
    # First message
    result1 = await agent.process_message(
        message="Message 1",
        system_prompt_name="test_prompt"
    )
    state_after_first = agent.messages.copy()
    
    # Second message should include history from first
    result2 = await agent.process_message(
        message="Message 2",
        system_prompt_name="test_prompt"
    )
    
    # Check that second message includes history
    assert len(agent.messages) > len(state_after_first)
    assert any(msg.content == "Message 1" for msg in agent.messages)
    assert any(msg.content == "Message 2" for msg in agent.messages)

@pytest.mark.asyncio
async def test_system_prompt_reuse(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test that system prompt is reused and not duplicated"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    
    # Multiple messages
    for i in range(3):  # Less than max_turns
        await agent.process_message(
            message=f"Message {i}",
            system_prompt_name="test_prompt"
        )
    
    # Check only one system prompt exists
    system_prompts = [msg for msg in agent.messages if msg.role == MessageRole.SYSTEM]
    assert len(system_prompts) == 1

@pytest.mark.asyncio
async def test_parser_exception_handling(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test handling of parser exceptions"""
    class FailingParser(MockParser):
        def parse(self, response: str) -> str:
            raise Exception("Parser error")
    
    agent = Agent(FailingParser, config_path=mock_config_no_tools)
    with pytest.raises(Exception) as exc_info:
        await agent.process_message(
            message="Test message",
            system_prompt_name="test_prompt"
        )
    assert "Parser error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_tool_output_disabled(mock_config_no_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test that tool outputs are not included when disabled"""
    agent = Agent(MockParser, config_path=mock_config_no_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt"
    )
    
    # Check that tool output was not added to history
    assert len(agent.messages) == 3  # system + user + assistant
    assert all(msg.role != MessageRole.TOOL for msg in agent.messages)

@pytest.mark.asyncio
async def test_tool_output_enabled(mock_config_with_tools, mock_system_prompt, mock_llm_provider, mock_llm_factory):
    """Test that tool outputs are included when enabled"""
    agent = Agent(MockParser, config_path=mock_config_with_tools)
    result = await agent.process_message(
        message="Test message",
        system_prompt_name="test_prompt"
    )
    
    # Check that tool output was added to history
    assert len(agent.messages) == 4  # system + user + assistant + tool
    assert any(msg.role == MessageRole.TOOL for msg in agent.messages)
    tool_message = next(msg for msg in agent.messages if msg.role == MessageRole.TOOL)
    assert tool_message.name == "command_output"
    assert tool_message.content == "Parsed: Test response 1"

@pytest.mark.asyncio
async def test_invalid_config(mock_config_no_tools):
    """Test handling of invalid config file"""
    # Corrupt the config file
    with open(mock_config_no_tools, 'w') as f:
        f.write("invalid: yaml: content")
    
    with pytest.raises(Exception) as exc_info:
        agent = Agent(MockParser, config_path=mock_config_no_tools)
    error_msg = str(exc_info.value)
    # Check for either the wrapped error message or the raw YAML error
    assert any(msg in error_msg for msg in [
        "Error loading config",
        "mapping values are not allowed",
        "invalid yaml"
    ]) 