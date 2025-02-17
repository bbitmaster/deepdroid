"""
Configuration manager for the agent.
Uses the singleton pattern to ensure only one config instance exists.
"""

import os
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Singleton configuration manager"""
    
    _instance = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to the config directory in the package
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(package_dir, 'config', 'agent_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                
            # Set up logging based on config
            log_level = self._config['agent'].get('log_level', 'INFO')
            logging.basicConfig(level=getattr(logging, log_level))
            
        except Exception as e:
            logger.error("Error loading config from {}: {}".format(config_path, e))
            raise Exception("Error loading config: " + str(e))
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        if self._config is None:
            self.load_config()
        return self._config
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration"""
        return self.config['llm']
    
    @property
    def agent_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        return self.config['agent']
    
    def get_system_prompt(self, prompt_name: str) -> str:
        """Load a system prompt from the configured directory"""
        prompt_dir = self.agent_config['system_prompt_dir']
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(package_dir, prompt_dir, f"{prompt_name}.txt")
        
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading system prompt {prompt_name}: {str(e)}")
            raise Exception("Error loading system prompt: " + str(e)) 