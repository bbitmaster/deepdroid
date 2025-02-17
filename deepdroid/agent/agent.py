"""
Main agent implementation that coordinates the parser, config manager, and LLM provider.
"""

import asyncio
import logging
import os
from typing import Optional, Type, Dict, Any, List
from deepdroid.parser import Parser
from deepdroid.agent.config import ConfigManager
from deepdroid.agent.llm_provider import (
    LLMProvider, 
    LLMProviderFactory, 
    LLMResponse, 
    Message, 
    MessageRole
)

logger = logging.getLogger(__name__)

class Agent:
    """
    Main agent class that coordinates the parser, config manager, and LLM provider.
    Uses dependency injection for the parser and configuration management.
    """
    
    def __init__(self, 
                 parser_class: Type[Parser],
                 config_path: Optional[str] = None,
                 **parser_kwargs):
        """
        Initialize the agent with a parser class and optional config path.
        
        Args:
            parser_class: The parser class to use for handling LLM responses
            config_path: Optional path to config file
            **parser_kwargs: Additional arguments to pass to the parser
        """
        # Initialize configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_config(config_path)
            
        # Initialize LLM provider
        llm_config = self.config_manager.llm_config
        self.llm = LLMProviderFactory.create(
            llm_config['provider'],
            llm_config
        )
        
        # Initialize parser with memory file from config
        parser_kwargs['memory_file'] = self.config_manager.agent_config['memory_file']
        self.parser = parser_class(**parser_kwargs)
        
        # Get retry settings from config
        self.max_retries = self.config_manager.agent_config['max_retries']
        self.retry_delay = self.config_manager.agent_config['retry_delay']
        
        # Initialize conversation history
        self.conversation_config = self.config_manager.agent_config['conversation']
        self.messages: List[Message] = []
        self.turn_count = 0
    
    def _get_total_chars(self) -> int:
        """Calculate total characters in conversation history"""
        return sum(len(msg.content) for msg in self.messages)
    
    def _should_reset_conversation(self) -> bool:
        """Check if we should reset the conversation based on config"""
        if self.turn_count >= self.conversation_config['max_turns']:
            logger.info("Resetting conversation: max turns reached")
            return True
        
        total_chars = self._get_total_chars()
        if total_chars >= self.conversation_config['max_chars']:
            logger.info(f"Resetting conversation: max characters reached ({total_chars} chars)")
            return True
        
        return False
    
    async def _reset_conversation(self) -> None:
        """Reset the conversation history, optionally with a summary"""
        if self.conversation_config['summarize_on_reset'] and self.messages:
            # Ask the LLM to summarize the conversation
            summary_prompt = Message(
                role=MessageRole.USER,
                content="Please provide a brief summary of our conversation so far, "
                       "focusing on the most important points and decisions made."
            )
            summary_response = await self.llm.generate(
                messages=self.messages + [summary_prompt],
                temperature=0.7
            )
            
            # Start fresh with just the summary
            self.messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="Previous conversation summary: " + summary_response.content
                )
            ]
            logger.info(f"Conversation reset with summary ({len(self.messages[0].content)} chars)")
        else:
            self.messages = []
            logger.info("Conversation reset without summary")
        
        self.turn_count = 0
    
    async def process_message(self,
                            message: str,
                            system_prompt_name: str,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None) -> str:
        """
        Process a message using the LLM and parser.
        
        Args:
            message: The input message to process
            system_prompt_name: Name of the system prompt file to use
            temperature: LLM temperature parameter
            max_tokens: Optional maximum tokens for response
            
        Returns:
            str: The processed result
        """
        logger.debug(f"Processing message ({len(message)} chars)")
        
        # Check if we need to reset the conversation
        if self._should_reset_conversation():
            logger.info("Resetting conversation due to limits")
            await self._reset_conversation()
        
        # If this is the first message or after reset, add the system prompt
        if not self.messages:
            system_prompt = self.config_manager.get_system_prompt(system_prompt_name)
            self.messages.append(Message(
                role=MessageRole.SYSTEM,
                content=system_prompt
            ))
            logger.debug(f"Added system prompt ({len(system_prompt)} chars)")
        
        # Add the user's message to the history
        self.messages.append(Message(
            role=MessageRole.USER,
            content=message
        ))
        logger.debug(f"Added user message ({len(message)} chars)")
        logger.debug(f"Current conversation state: {len(self.messages)} messages, {self._get_total_chars()} total chars")
        
        # Try to get LLM response with retries
        response = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempting LLM generation (attempt {attempt + 1}/{self.max_retries})")
                response = await self.llm.generate(
                    messages=self.messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                logger.debug(f"LLM generation successful ({len(response.content)} chars)")
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"All LLM generation attempts failed: {str(e)}")
                    raise
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(self.retry_delay)
        
        # Update conversation history with the new messages
        self.messages = response.messages
        logger.debug(f"Updated conversation history (total {self._get_total_chars()} chars)")
        
        # Parse and execute the LLM response
        result = self.parser.parse(response.content)
        
        # Optionally add tool output to conversation history
        if (result and self.conversation_config['include_tool_outputs']):
            self.messages.append(Message(
                role=MessageRole.TOOL,
                content=result,
                name="command_output"
            ))
            logger.debug(f"Added tool output ({len(result)} chars)")
        
        self.turn_count += 1
        logger.info(f"Completed turn {self.turn_count} ({len(self.messages)} messages in context)")
        return result
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.parser:
            self.parser.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.cleanup()
        return False  # Don't suppress exceptions

# Example usage:
async def run_agent(message: str, parser_class: Type[Parser], **parser_kwargs):
    async with Agent(parser_class, **parser_kwargs) as agent:
        result = await agent.process_message(
            message=message,
            system_prompt_name="codebase_improver"  # or other prompt name
        )
        return result 