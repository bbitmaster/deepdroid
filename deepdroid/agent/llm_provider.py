"""
LLM provider interface and implementations.
Uses the strategy pattern to support different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
import aiohttp
import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# HTTP status codes that should trigger a retry
RETRY_STATUS_CODES = {
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Cloudflare: Unknown Error
    524,  # Cloudflare: A Timeout Occurred
}

class MessageRole(Enum):
    """Enum for message roles in the conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # For tool/function outputs

@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool messages

@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    messages: List[Message]  # Full conversation history

def format_chat_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Format messages into the standard chat API format"""
    return [
        {
            'role': msg.role.value,
            'content': msg.content,
            **({"name": msg.name} if msg.name else {})
        }
        for msg in messages
    ]

def parse_chat_response(response_data: Dict[str, Any], original_messages: List[Message]) -> LLMResponse:
    """Parse a chat API response into a standardized format"""
    new_message = Message(
        role=MessageRole.ASSISTANT,
        content=response_data['choices'][0]['message']['content']
    )
    updated_messages = original_messages + [new_message]
    
    return LLMResponse(
        content=new_message.content,
        model=response_data['model'],
        usage=response_data['usage'],
        raw_response=response_data,
        messages=updated_messages
    )

class LLMProviderError(Exception):
    """Base class for LLM provider errors"""
    pass

class RetryableError(LLMProviderError):
    """Error that can be retried"""
    pass

class NonRetryableError(LLMProviderError):
    """Error that should not be retried"""
    pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, 
                      messages: List[Message],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages in the conversation
            temperature: Controls randomness in the response
            max_tokens: Optional maximum tokens for response
            
        Returns:
            LLMResponse containing the generated response and conversation history
        """
        pass

    async def _make_request(self,
                          url: str,
                          headers: Dict[str, str],
                          data: Dict[str, Any],
                          timeout: int,
                          max_retries: int,
                          retry_delay: int) -> Dict[str, Any]:
        """Make an HTTP request with retries"""
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        
                        error_text = await response.text()
                        if response.status in RETRY_STATUS_CODES:
                            raise RetryableError(f"HTTP {response.status}: {error_text}")
                        else:
                            raise NonRetryableError(f"HTTP {response.status}: {error_text}")
                            
            except asyncio.TimeoutError:
                last_error = RetryableError(f"Request timed out after {timeout} seconds")
            except aiohttp.ClientError as e:
                last_error = RetryableError(f"Network error: {str(e)}")
            except RetryableError as e:
                last_error = e
            except Exception as e:
                if isinstance(e, NonRetryableError):
                    raise
                last_error = RetryableError(f"Unexpected error: {str(e)}")
            
            attempt += 1
            if attempt < max_retries:
                delay = retry_delay * (1.5 ** (attempt - 1))  # Exponential backoff
                logger.warning(f"Attempt {attempt} failed: {str(last_error)}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            
        raise last_error or RetryableError("Max retries exceeded")

class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.api_key = config['api_key'] or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in config or environment")
        
        self.model = config['default_model']
        self.timeout = config['timeout']
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
    
    async def generate(self,
                      messages: List[Message],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate a response using OpenRouter API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/yourusername/deepdroid',  # Replace with your repo
        }
        
        data = {
            'model': self.model,
            'messages': format_chat_messages(messages),
            'temperature': temperature,
        }
        if max_tokens:
            data['max_tokens'] = max_tokens
            
        try:
            result = await self._make_request(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                data=data,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
            return parse_chat_response(result, messages)
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.api_key = config['api_key'] or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        self.model = config['default_model']
        self.timeout = config['timeout']
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
    
    async def generate(self,
                      messages: List[Message],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate a response using OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        data = {
            'model': self.model,
            'messages': format_chat_messages(messages),
            'temperature': temperature,
        }
        if max_tokens:
            data['max_tokens'] = max_tokens
            
        try:
            result = await self._make_request(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                data=data,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
            return parse_chat_response(result, messages)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create(provider: str, config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM provider instance based on the provider name"""
        providers = {
            'openrouter': OpenRouterProvider,
            'openai': OpenAIProvider
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        return providers[provider](config[provider]) 