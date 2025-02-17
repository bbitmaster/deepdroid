"""
Simple CLI tool for testing LLM API connections.
"""

import asyncio
import argparse
import logging
from typing import Optional
from deepdroid.agent.llm_provider import (
    LLMProviderFactory,
    Message,
    MessageRole
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test LLM API connections by sending a prompt and getting a response"
    )
    parser.add_argument(
        "prompt",
        help="The prompt to send to the LLM"
    )
    parser.add_argument(
        "-s", "--system",
        help="Optional system prompt to set context",
        default=None
    )
    parser.add_argument(
        "--config",
        help="Path to config file (default: agent_config.yaml)",
        default=None
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "openai"],
        default="openrouter",
        help="Which LLM provider to use"
    )
    parser.add_argument(
        "--model",
        help="Override the model specified in config",
        default=None
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0 to 1.0)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output including full API response"
    )
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s'  # Simple format for CLI tool
    )
    logger = logging.getLogger(__name__)
    
    # Build messages list
    messages = []
    if args.system:
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content=args.system
        ))
    messages.append(Message(
        role=MessageRole.USER,
        content=args.prompt
    ))
    
    try:
        # Create provider
        from deepdroid.agent.config import ConfigManager
        config = ConfigManager()
        if args.config:
            config.load_config(args.config)
        
        provider_config = config.config['llm'][args.provider]
        if args.model:
            provider_config['default_model'] = args.model
            
        provider = LLMProviderFactory.create(args.provider, config.config['llm'])
        
        # Generate response
        logger.debug("Sending request to %s...", args.provider)
        response = await provider.generate(
            messages=messages,
            temperature=args.temperature
        )
        
        # Print response
        if args.verbose:
            logger.debug("\nFull response:")
            logger.debug("Model: %s", response.model)
            logger.debug("Usage: %s", response.usage)
            logger.debug("\nResponse content:")
            
        print(response.content)
            
    except Exception as e:
        logger.error("Error: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        exit(1) 