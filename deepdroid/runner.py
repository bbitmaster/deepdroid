"""
Task runner implementation for iterative codebase improvement.
"""

import asyncio
import logging
from typing import Optional, List
from deepdroid.agent.agent import Agent
from deepdroid.parser import Parser
from deepdroid.parsers.codebase_improver import CodebaseImproverParser

logger = logging.getLogger(__name__)

class TaskRunner:
    """
    Task runner that coordinates between the agent and parser to iteratively improve the codebase.
    
    The runner maintains the conversation state and handles:
    1. Message passing between agent and parser
    2. Context window management
    3. System prompt updates
    4. Continuous improvement loop
    """
    
    def __init__(self, agent: Agent, parser: Parser):
        """Initialize with an agent and parser instance"""
        self.agent = agent
        self.parser = parser
        
        # Ensure parser is a CodebaseImproverParser for system prompt updates
        if not isinstance(parser, CodebaseImproverParser):
            raise ValueError("Parser must be an instance of CodebaseImproverParser")
        
        logger.info("TaskRunner initialized with agent and parser")
    
    async def run(self, initial_message: str = "Analyze the codebase and suggest improvements.", 
                 max_iterations: Optional[int] = None) -> None:
        """
        Run the improvement loop.
        
        Args:
            initial_message: The first message to send to the agent
            max_iterations: Optional maximum number of iterations (None for unlimited)
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
                logger.debug(f"Current message length: {len(current_message)} chars")
                
                # Get the current system prompt from parser (with updated memories)
                system_prompt = self.parser.get_system_prompt()
                if not system_prompt:
                    logger.error("System prompt not set in parser")
                    raise ValueError("System prompt not set in parser")
                
                # Process the message with the agent
                logger.debug("Sending message to agent for processing")
                print(f"DEBUG: About to call process_message with message: {current_message}")
                result = await self.agent.process_message(
                    message=current_message,
                    system_prompt_name="codebase_improver"
                )
                print(f"DEBUG: Returned result from process_message: {result}")
                logger.debug(f"Received result from agent ({len(result)} chars)")
                
                # Parse the result and get feedback
                logger.debug("Parsing result with CodebaseImprover")
                feedback = self.parser.parse(result)
                print(f"DEBUG: Feedback from parser: {feedback}")
                logger.debug(f"Received feedback from parser ({len(feedback)} chars if any)")
                
                # Use the feedback as the next message
                if not feedback.strip():
                    logger.info("No feedback received, ending improvement loop")
                    break
                    
                current_message = (
                    "Previous actions have been completed. Here is the output:\n\n"
                    f"{feedback}\n\n"
                    "Please analyze this output and determine the next improvements to make."
                )
                
                iteration += 1
                logger.info(f"Completed iteration {iteration}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt, cleaning up...")
        except Exception as e:
            logger.error(f"Error in improvement loop: {str(e)}")
            raise
        finally:
            # Ensure cleanup
            logger.info("Cleaning up resources...")
            self.agent.cleanup()
            self.parser.cleanup()
            logger.info("Cleanup complete")
            
    @classmethod
    async def create_and_run(cls, 
                            config_path: Optional[str] = None,
                            initial_message: str = "Analyze the codebase and suggest improvements.",
                            max_iterations: Optional[int] = None) -> None:
        """
        Create a runner with default agent and parser and run it.
        
        Args:
            config_path: Optional path to config file
            initial_message: Initial message to start with
            max_iterations: Optional maximum number of iterations
        """
        logger.info("Creating new TaskRunner instance")
        if config_path:
            logger.info(f"Using config from: {config_path}")
        
        async with Agent(CodebaseImproverParser, config_path=config_path) as agent:
            logger.debug("Created Agent")
            runner = cls(agent, agent.parser)
            await runner.run(initial_message, max_iterations) 