"""
Core improvement loop implementation.
"""

import logging
from typing import Optional
from deepdroid.agent.agent import Agent
from deepdroid.parsers.codebase_improver import CodebaseImproverParser

logger = logging.getLogger(__name__)

async def run_improvement_loop(agent: Agent,
                             parser: CodebaseImproverParser,
                             initial_message: str,
                             max_iterations: Optional[int] = None) -> None:
    """
    Run the main improvement loop.
    
    Args:
        agent: The agent to use for processing messages
        parser: The parser to use for handling agent responses
        initial_message: The first message to send to the agent
        max_iterations: Optional maximum number of iterations
        
    Raises:
        ValueError: If the system prompt is not set in the parser
    """
    iteration = 0
    current_message = initial_message
    
    logger.info("Starting improvement loop")
    if max_iterations:
        logger.info(f"Will run for maximum of {max_iterations} iterations")
    else:
        logger.info("Will run until no more improvements found")
    
    try:
        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Reached maximum iterations ({max_iterations})")
                break
            
            logger.info(f"Starting iteration {iteration + 1}")
            
            # Get the current system prompt from parser
            system_prompt = parser.get_system_prompt()
            if not system_prompt:
                logger.error("System prompt not set in parser")
                raise ValueError("System prompt not set in parser")
            
            # Process the message with the agent
            logger.debug("Sending message to agent for processing")
            result = await agent.process_message(
                message=current_message,
                system_prompt_name="codebase_improver"
            )
            
            # Check if we got any output
            if not result or not result.strip():
                logger.info("No output received, ending improvement loop")
                break
                
            # Use the result as the next message
            current_message = (
                "Previous actions have been completed. Here is the output:\n\n"
                f"{result}\n\n"
                "Please analyze this output and determine the next improvements to make."
            )
            
            iteration += 1
            logger.info(f"Completed iteration {iteration}")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt, cleaning up...")
    except Exception as e:
        logger.error(f"Error in improvement loop: {str(e)}")
        raise 