"""
Main entrypoint for deepdroid package.
"""

import asyncio
import argparse
import logging
from deepdroid.agent.agent import Agent
from deepdroid.parsers.codebase_improver import CodebaseImproverParser
from deepdroid.improver import run_improvement_loop

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DeepDroid - An LLM-powered agent framework for improving codebases"
    )
    parser.add_argument(
        "--config",
        help="Path to config file",
        default=None
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of improvement iterations",
        default=None
    )
    parser.add_argument(
        "--initial-message",
        help="Initial message to start the improvement process",
        default="Analyze the codebase and suggest improvements."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run the agent
        async with Agent(CodebaseImproverParser, config_path=args.config) as agent:
            logger.info("Created agent with CodebaseImproverParser")
            
            # Ensure we have a CodebaseImproverParser
            if not isinstance(agent.parser, CodebaseImproverParser):
                raise ValueError("Agent must use a CodebaseImproverParser")
            
            await run_improvement_loop(
                agent=agent,
                parser=agent.parser,
                initial_message=args.initial_message,
                max_iterations=args.max_iterations
            )
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceived interrupt, exiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 