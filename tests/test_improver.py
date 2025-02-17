"""
Tests for the improvement loop implementation.
"""

import pytest
import logging
from unittest.mock import AsyncMock, Mock, patch
from deepdroid.improver import run_improvement_loop
from deepdroid.agent.agent import Agent
from deepdroid.parsers.codebase_improver import CodebaseImproverParser

@pytest.fixture
def mock_agent():
    """Create a mock agent"""
    agent = AsyncMock(spec=Agent)
    agent.process_message = AsyncMock(return_value="Mock LLM Response")
    return agent

@pytest.fixture
def mock_parser():
    """Create a mock parser"""
    parser = Mock(spec=CodebaseImproverParser)
    parser.get_system_prompt = Mock(return_value="Mock System Prompt")
    parser.parse = Mock(return_value="Mock Parser Feedback")
    return parser

@pytest.mark.asyncio
async def test_basic_improvement_loop(mock_agent, mock_parser):
    """Test basic improvement loop execution"""
    await run_improvement_loop(
        agent=mock_agent,
        parser=mock_parser,
        initial_message="Test message",
        max_iterations=1
    )
    
    # Verify interactions
    mock_parser.get_system_prompt.assert_called_once()
    mock_agent.process_message.assert_called_once_with(
        message="Test message",
        system_prompt_name="codebase_improver"
    )
    mock_parser.parse.assert_called_once_with("Mock LLM Response")

@pytest.mark.asyncio
async def test_improvement_loop_no_feedback(mock_agent, mock_parser):
    """Test loop termination when no feedback received"""
    mock_parser.parse = Mock(return_value="")  # Empty feedback
    
    await run_improvement_loop(
        agent=mock_agent,
        parser=mock_parser,
        initial_message="Test message"
    )
    
    # Should only run once due to empty feedback
    mock_agent.process_message.assert_called_once()

@pytest.mark.asyncio
async def test_improvement_loop_max_iterations(mock_agent, mock_parser):
    """Test loop respects max iterations"""
    await run_improvement_loop(
        agent=mock_agent,
        parser=mock_parser,
        initial_message="Test message",
        max_iterations=3
    )
    
    assert mock_agent.process_message.call_count == 3
    assert mock_parser.parse.call_count == 3

@pytest.mark.asyncio
async def test_improvement_loop_system_prompt_missing(mock_agent, mock_parser):
    """Test handling of missing system prompt"""
    mock_parser.get_system_prompt = Mock(return_value=None)
    
    with pytest.raises(ValueError) as exc_info:
        await run_improvement_loop(
            agent=mock_agent,
            parser=mock_parser,
            initial_message="Test message"
        )
    assert "System prompt not set" in str(exc_info.value)

@pytest.mark.asyncio
async def test_improvement_loop_error_handling(mock_agent, mock_parser):
    """Test error handling in loop"""
    mock_agent.process_message = AsyncMock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception) as exc_info:
        await run_improvement_loop(
            agent=mock_agent,
            parser=mock_parser,
            initial_message="Test message"
        )
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_improvement_loop_keyboard_interrupt(mock_agent, mock_parser, caplog):
    """Test handling of keyboard interrupt"""
    mock_agent.process_message = AsyncMock(side_effect=KeyboardInterrupt)
    
    with caplog.at_level(logging.INFO):
        await run_improvement_loop(
            agent=mock_agent,
            parser=mock_parser,
            initial_message="Test message"
        )
    
    assert "Received interrupt, cleaning up..." in caplog.text 