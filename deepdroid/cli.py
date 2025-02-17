"""
Command line interface for deepdroid.
"""

import asyncio
import click
import logging
from typing import Optional

from deepdroid.agent.llm_provider import (
    LLMProviderFactory,
    Message,
    MessageRole
)
from deepdroid.agent.config import ConfigManager
from deepdroid.agent.agent import Agent
from deepdroid.parsers.codebase_improver import CodebaseImproverParser
from deepdroid.improver import run_improvement_loop

# Set up logging
logging.basicConfig(format='%(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.option('--debug/--no-debug', default=False, help="Enable debug logging")
def cli(debug):
    """DeepDroid - A minimal framework for building LLM-powered agents."""
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

@cli.command()
@click.argument('prompt')
@click.option('-s', '--system', help="Optional system prompt")
@click.option('--config', help="Path to config file")
@click.option('--provider', type=click.Choice(['openrouter', 'openai']), default='openrouter')
@click.option('--model', help="Override the model specified in config")
@click.option('--temperature', type=float, default=0.7, help="Temperature (0.0 to 1.0)")
@click.option('-v', '--verbose', is_flag=True, help="Show verbose output")
def test(prompt, system, config, provider, model, temperature, verbose):
    """Test the LLM connection with a simple prompt."""
    async def run():
        # Build messages
        messages = []
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        try:
            # Set up provider
            config_mgr = ConfigManager()
            if config:
                config_mgr.load_config(config)
            
            provider_config = config_mgr.config['llm'][provider]
            if model:
                provider_config['default_model'] = model
                
            llm = LLMProviderFactory.create(provider, config_mgr.config['llm'])
            
            # Generate response
            if verbose:
                click.echo(f"Sending request to {provider}...")
            response = await llm.generate(messages=messages, temperature=temperature)
            
            # Print response
            if verbose:
                click.echo("\nFull response:")
                click.echo(f"Model: {response.model}")
                click.echo(f"Usage: {response.usage}")
                click.echo("\nResponse content:")
            
            click.echo(response.content)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
    
    asyncio.run(run())

@cli.command()
@click.option('--config', help="Path to config file")
@click.option('--max-iterations', type=int, help="Maximum number of improvement iterations")
@click.option('--message', default="Analyze the codebase and suggest improvements.",
              help="Initial message to start the improvement process")
def improve(config, max_iterations, message):
    """Run the codebase improvement agent."""
    async def run():
        try:
            async with Agent(CodebaseImproverParser, config_path=config) as agent:
                if not isinstance(agent.parser, CodebaseImproverParser):
                    raise click.UsageError("Agent must use a CodebaseImproverParser")
                
                await run_improvement_loop(
                    agent=agent,
                    parser=agent.parser,
                    initial_message=message,
                    max_iterations=max_iterations
                )
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
    
    asyncio.run(run())

if __name__ == '__main__':
    cli() 