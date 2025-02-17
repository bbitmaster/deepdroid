"""
Tests for the TaskRunner class.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock, ANY
from typing import List, Dict, Any, Optional

from deepdroid.runner import TaskRunner
from deepdroid.agent.agent import Agent
from deepdroid.parsers.codebase_improver import CodebaseImproverParser
from deepdroid.agent.llm_provider import Message, MessageRole

@pytest.fixture
def mock_agent():
    """Create a mock agent"""
    agent = AsyncMock(spec=Agent)
    agent.process_message = AsyncMock(return_value="Mock LLM Response")
    agent.cleanup = Mock()
    return agent

@pytest.fixture
def mock_parser():
    """Create a mock parser with CodebaseImproverParser interface"""
    parser = Mock(spec=CodebaseImproverParser)
    parser.parse = Mock(return_value="Mock Parser Feedback")
    parser.get_system_prompt = Mock(return_value="Mock System Prompt")
    parser.cleanup = Mock()
    return parser

@pytest.mark.asyncio
async def test_runner_initialization(mock_agent, mock_parser):
    """Test basic runner initialization"""
    runner = TaskRunner(mock_agent, mock_parser)
    assert runner.agent == mock_agent
    assert runner.parser == mock_parser

@pytest.mark.asyncio
async def test_runner_with_invalid_parser(mock_agent):
    """Test runner initialization with non-CodebaseImproverParser"""
    invalid_parser = Mock()  # Not a CodebaseImproverParser
    with pytest.raises(ValueError) as exc_info:
        TaskRunner(mock_agent, invalid_parser)
    assert "must be an instance of CodebaseImproverParser" in str(exc_info.value)

@pytest.mark.asyncio
async def test_basic_improvement_loop(mock_agent, mock_parser):
    """Test basic improvement loop execution"""
    runner = TaskRunner(mock_agent, mock_parser)
    
    # Run for exactly one iteration
    await runner.run(max_iterations=1)
    
    # Verify interactions
    mock_parser.get_system_prompt.assert_called_once()
    mock_agent.process_message.assert_called_once()
    mock_parser.parse.assert_called_once_with("Mock LLM Response")
    
    # Verify cleanup
    mock_agent.cleanup.assert_called_once()
    mock_parser.cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_improvement_loop_no_feedback(mock_agent, mock_parser):
    """Test improvement loop termination when no feedback received"""
    mock_parser.parse = Mock(return_value="")  # Empty feedback
    runner = TaskRunner(mock_agent, mock_parser)
    
    await runner.run()
    
    # Should only run once due to empty feedback
    mock_agent.process_message.assert_called_once()

@pytest.mark.asyncio
async def test_improvement_loop_max_iterations(mock_agent, mock_parser):
    """Test improvement loop respects max iterations"""
    runner = TaskRunner(mock_agent, mock_parser)
    
    await runner.run(max_iterations=3)
    
    assert mock_agent.process_message.call_count == 3
    assert mock_parser.parse.call_count == 3

@pytest.mark.asyncio
async def test_improvement_loop_system_prompt_missing(mock_agent, mock_parser):
    """Test handling of missing system prompt"""
    mock_parser.get_system_prompt = Mock(return_value=None)
    runner = TaskRunner(mock_agent, mock_parser)
    
    with pytest.raises(ValueError) as exc_info:
        await runner.run()
    assert "System prompt not set" in str(exc_info.value)

@pytest.mark.asyncio
async def test_improvement_loop_error_handling(mock_agent, mock_parser):
    """Test error handling in improvement loop"""
    mock_agent.process_message = AsyncMock(side_effect=Exception("Test error"))
    runner = TaskRunner(mock_agent, mock_parser)
    
    with pytest.raises(Exception) as exc_info:
        await runner.run()
    
    assert "Test error" in str(exc_info.value)
    # Verify cleanup still occurred
    mock_agent.cleanup.assert_called_once()
    mock_parser.cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_improvement_loop_keyboard_interrupt(mock_agent, mock_parser):
    """Test handling of keyboard interrupt"""
    mock_agent.process_message = AsyncMock(side_effect=KeyboardInterrupt)
    runner = TaskRunner(mock_agent, mock_parser)
    
    await runner.run()
    
    # Verify cleanup occurred
    mock_agent.cleanup.assert_called_once()
    mock_parser.cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_create_and_run_method():
    """Test the create_and_run class method"""
    # Mock the Agent class
    mock_agent = AsyncMock()
    mock_agent.parser = Mock(spec=CodebaseImproverParser)  # Use spec to support isinstance
    mock_agent.parser.get_system_prompt.return_value = "Mock system prompt"
    mock_agent.parser.parse.return_value = "Mock parser feedback"  # Return a string for len()
    mock_agent.__aenter__.return_value = mock_agent
    mock_agent.__aexit__.return_value = None
    mock_agent.process_message.return_value = "Test response"
    
    # Create a real TaskRunner for testing
    runner = TaskRunner(mock_agent, mock_agent.parser)
    
    with patch('deepdroid.runner.Agent', return_value=mock_agent), \
         patch.object(TaskRunner, '__new__', return_value=runner):
        
        # Test with default parameters
        await TaskRunner.create_and_run()
        
        # Verify Agent was created with correct parameters
        mock_agent.assert_called_once_with(
            CodebaseImproverParser,
            config_path=None
        )
        
        # Verify run was called with default parameters
        assert runner.agent == mock_agent
        assert runner.parser == mock_agent.parser
        
        # Test with custom parameters
        mock_agent.reset_mock()
        
        config_path = "test_config.yaml"
        initial_message = "Custom message"
        max_iterations = 5
        
        await TaskRunner.create_and_run(
            config_path=config_path,
            initial_message=initial_message,
            max_iterations=max_iterations
        )
        
        # Verify Agent was created with custom config
        mock_agent.assert_called_once_with(
            CodebaseImproverParser,
            config_path=config_path
        )