"""
Parser implementation for the codebase improver agent.
This parser handles the execution of commands from an LLM aimed at improving codebases.
"""

from deepdroid.parser import Parser
import re
import subprocess
import os
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import logging
from deepdroid.agent.config import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single entry in the agent's memory"""
    id: int
    content: str

class CodebaseImproverParser(Parser):
    """
    Parser implementation for the codebase improver agent.
    Handles execution of commands for analyzing and improving codebases.
    """
    
    def __init__(self, memory_file: str = 'agent_memory.txt', max_memory: int = 100):
        super().__init__()
        self.max_memory = max_memory
        self.memory_file = memory_file
        self.memories: List[MemoryEntry] = []
        self.system_prompt: Optional[str] = None
        self._load_memories()
        self._load_system_prompt()
    
    def _load_system_prompt(self) -> None:
        """Load the codebase improver system prompt"""
        try:
            config = ConfigManager()
            system_prompt_dir = config.agent_config['system_prompt_dir']
            prompt_path = os.path.join(system_prompt_dir, "codebase_improver.txt")
            with open(prompt_path, 'r') as f:
                self.set_system_prompt(f.read())
        except Exception as e:
            logger.error(f"Error loading system prompt: {str(e)}")
            raise
    
    def _load_memories(self) -> None:
        """Load memories from persistent storage if it exists"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            id_str, content = line.strip().split(':', 1)
                            self.memories.append(MemoryEntry(int(id_str), content.strip()))
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
    
    def _save_memories(self) -> None:
        """Save memories to persistent storage"""
        try:
            with open(self.memory_file, 'w') as f:
                for memory in self.memories:
                    f.write(f"{memory.id}: {memory.content}\n")
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")

    def _renumber_memories(self) -> None:
        """Renumber all memories sequentially starting from 1"""
        for i, memory in enumerate(self.memories, 1):
            memory.id = i

    def _update_memory_section(self) -> None:
        """Update the memory section in the system prompt"""
        if not self.system_prompt:
            return

        # Generate memory contents
        memory_lines = []
        if not self.memories:
            memory_lines.append("1: blank - no memories stored yet")
        else:
            for memory in self.memories:
                memory_lines.append(f"{memory.id}: {memory.content}")
        
        memory_content = "\n".join(memory_lines)
        
        # Calculate buffer usage
        usage_percent = (len(self.memories) / self.max_memory) * 100
        usage_str = f"Memory buffer usage: {len(self.memories)}/{self.max_memory} entries ({usage_percent:.0f}%)"
        
        # Update memory contents section
        self.system_prompt = re.sub(
            r'<memory_contents>\n.*?\n</memory_contents>',
            f'<memory_contents>\n{memory_content}\n</memory_contents>',
            self.system_prompt,
            flags=re.DOTALL
        )
        
        # Update buffer usage line
        self.system_prompt = re.sub(
            r'Memory buffer usage:.*?\n',
            f'{usage_str}\n',
            self.system_prompt
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and initialize memory section"""
        self.system_prompt = prompt
        self._update_memory_section()

    def get_system_prompt(self) -> Optional[str]:
        """Get the current system prompt with updated memory section"""
        return self.system_prompt

    def cleanup(self) -> None:
        """Ensure memories are saved during cleanup"""
        self._save_memories()

    def parse(self, response: str) -> str:
        """
        Parse and execute LLM commands from the response.
        
        Args:
            response: The LLM's response containing XML-formatted commands
            
        Returns:
            str: Results of executing the commands
        """
        logger.debug(f"Parsing response ({len(response)} chars)")
        output_lines = []
        
        # Convert the output to an XML structure for easier parsing
        xml_string = f"<root>{response}</root>"
        
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error(f"Error parsing LLM output as XML: {str(e)}")
            return f"Error parsing LLM output as XML: {str(e)}"

        # Process each command in order
        command_count = 0
        for element in root:
            command_count += 1
            logger.debug(f"Processing command {command_count}: {element.tag}")
            result = self._execute_command(element)
            if result:
                output_lines.append(result)

        output = "\n".join(output_lines)
        logger.debug(f"Completed parsing {command_count} commands, produced {len(output)} chars of output")
        return output

    def _execute_command(self, element: ET.Element) -> Optional[str]:
        """Execute a single XML command element"""
        handlers = {
            'thinking': self._handle_thinking,
            'memory_append': self._handle_memory_append,
            'memory_remove': self._handle_memory_remove,
            'bash': self._handle_bash,
            'read_file': self._handle_read_file,
            'write_file': self._handle_write_file,
            'patch': self._handle_patch,
            'backup_file': self._handle_backup,
            'run_tests': self._handle_tests
        }
        
        handler = handlers.get(element.tag)
        if handler:
            return handler(element)
        return f"Unknown command tag: {element.tag}"

    def _handle_thinking(self, element: ET.Element) -> None:
        """Handle thinking tag - no action needed"""
        return None

    def _handle_memory_append(self, element: ET.Element) -> str:
        """Handle memory append command"""
        content = element.text.strip()
        logger.debug(f"Appending memory ({len(content)} chars)")
        if len(self.memories) >= self.max_memory:
            logger.info("Memory limit reached, removing oldest memory")
            self.memories.pop(0)  # Remove oldest memory
        new_memory = MemoryEntry(0, content)  # Temporary ID
        self.memories.append(new_memory)
        self._renumber_memories()  # Renumber all memories
        self._save_memories()
        self._update_memory_section()  # Update system prompt after adding memory
        logger.debug(f"Memory appended, now have {len(self.memories)}/{self.max_memory} memories")
        return f"Memory appended: {new_memory.content}"

    def _handle_memory_remove(self, element: ET.Element) -> str:
        """Handle memory remove command"""
        try:
            memory_id = int(element.get('id'))
            self.memories = [m for m in self.memories if m.id != memory_id]
            self._renumber_memories()  # Renumber remaining memories
            self._save_memories()
            self._update_memory_section()  # Update system prompt after removing memory
            return f"Memory {memory_id} removed"
        except ValueError:
            return f"Invalid memory ID: {element.get('id')}"

    def _handle_bash(self, element: ET.Element) -> str:
        """Handle bash command execution"""
        command = element.text.strip()
        logger.info(f"Executing command: {command}")
        try:
            cmd_output = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = [f"Command output:\n{cmd_output.stdout}"]
            if cmd_output.stderr:
                output.append(f"Error output:\n{cmd_output.stderr}")
                logger.warning(f"Command produced error output: {cmd_output.stderr}")
            else:
                logger.debug(f"Command completed successfully: {len(cmd_output.stdout)} chars of output")
            return "\n".join(output)
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after 30 seconds: {command}")
            return "Command timed out after 30 seconds"
        except Exception as e:
            logger.error(f"Error executing command '{command}': {str(e)}")
            return f"Error executing command: {str(e)}"

    def _handle_read_file(self, element: ET.Element) -> str:
        """Handle file read command"""
        filename = element.get('filename')
        try:
            with open(filename, 'r') as f:
                content = f.read()
            numbered_content = '\n'.join(f"{i+1}: {line}" 
                                       for i, line in enumerate(content.splitlines()))
            return f"Contents of {filename}:\n{numbered_content}"
        except Exception as e:
            return f"Error reading file {filename}: {str(e)}"

    def _handle_write_file(self, element: ET.Element) -> str:
        """Handle file write command"""
        filename = element.get('filename')
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write(element.text)
            return f"File {filename} written successfully"
        except Exception as e:
            return f"Error writing file {filename}: {str(e)}"

    def _handle_patch(self, element: ET.Element) -> str:
        """Handle patch command"""
        filename = element.get('filename')
        try:
            with open('temp.patch', 'w') as f:
                f.write(element.text)
            
            result = subprocess.run(
                ['patch', filename, 'temp.patch'],
                capture_output=True,
                text=True
            )
            
            os.remove('temp.patch')  # Clean up
            
            if result.returncode == 0:
                return f"Successfully patched {filename}"
            return f"Error patching {filename}: {result.stderr}"
        except Exception as e:
            return f"Error applying patch to {filename}: {str(e)}"

    def _handle_backup(self, element: ET.Element) -> str:
        """Handle file backup command"""
        filename = element.get('filename')
        try:
            backup_name = f"{filename}.bak"
            with open(filename, 'r') as src, open(backup_name, 'w') as dst:
                dst.write(src.read())
            return f"Created backup: {backup_name}"
        except Exception as e:
            return f"Error creating backup of {filename}: {str(e)}"

    def _handle_tests(self, element: ET.Element) -> str:
        """Handle test execution command"""
        path = element.get('path', '.')
        try:
            result = subprocess.run(
                ['pytest', path],
                capture_output=True,
                text=True
            )
            output = [f"Test results:\n{result.stdout}"]
            if result.stderr:
                output.append(f"Test errors:\n{result.stderr}")
            return "\n".join(output)
        except Exception as e:
            return f"Error running tests: {str(e)}"

# Example usage:
def process_llm_response(llm_output: str) -> str:
    """Process the LLM's response and execute any commands"""
    executor = CodebaseImproverParser()
    return executor.parse(llm_output)

# Example:
if __name__ == "__main__":
    sample_output = '''
    <thinking>First, let's check the current directory structure</thinking>
    <bash>ls -la</bash>
    <memory_append>task: Initial directory structure analysis</memory_append>
    '''
    result = process_llm_response(sample_output)
    print(result)