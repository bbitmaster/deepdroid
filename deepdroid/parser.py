"""
Base parser class for deepdroid agents.
Defines the interface and common functionality for all parser implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Parser(ABC):
    """
    Abstract base class for all parsers in the deepdroid framework.
    
    A parser is responsible for:
    1. Parsing LLM responses into executable commands
    2. Executing those commands safely
    3. Providing feedback about the execution
    4. Maintaining any necessary state
    
    Implementations should:
    1. Override the parse() method to handle specific response formats
    2. Implement appropriate safety checks
    3. Handle errors gracefully
    4. Provide clear feedback
    """
    
    def __init__(self):
        """Initialize the parser with any necessary state"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, response: str) -> str:
        """
        Parse and execute commands from an LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            str: Results and feedback from executing the commands
            
        This method should:
        1. Parse the response into commands
        2. Execute commands safely
        3. Return results and feedback
        4. Handle errors appropriately
        """
        raise NotImplementedError("Parser implementations must override parse()")
    
    def validate_command(self, command: Any) -> bool:
        """
        Validate a command before execution.
        
        Args:
            command: The command to validate
            
        Returns:
            bool: True if the command is valid, False otherwise
            
        Override this method to implement command-specific validation.
        """
        return True
    
    def execute_command(self, command: Any) -> Optional[str]:
        """
        Execute a single command safely.
        
        Args:
            command: The command to execute
            
        Returns:
            Optional[str]: Result of the command execution, if any
            
        Override this method to implement command-specific execution.
        """
        return None
    
    def handle_error(self, error: Exception, context: str = "") -> str:
        """
        Handle errors during parsing or execution.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            str: Error message suitable for returning to the agent
        """
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg)
        return error_msg
    
    def cleanup(self) -> None:
        """
        Perform any necessary cleanup.
        
        Override this method to implement cleanup of resources.
        Called when the parser is done being used.
        """
        pass
    
    def __enter__(self):
        """Support for use in context managers"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup is called when used as context manager"""
        self.cleanup()
        return False  # Don't suppress exceptions 