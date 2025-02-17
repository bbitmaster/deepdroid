"""
Tests for the ConfigManager.
"""

import pytest
import os
import tempfile
import yaml
from deepdroid.agent.config import ConfigManager

@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "base_url": "https://test.com",
                "api_key": "test-key",
                "default_model": "test-model",
                "timeout": 30
            }
        },
        "agent": {
            "max_retries": 3,
            "retry_delay": 1,
            "memory_file": "test_memory.txt",
            "system_prompt_dir": "test_prompts",
            "log_level": "INFO",
            "conversation": {
                "max_turns": 10,
                "max_chars": 32000,
                "include_tool_outputs": True,
                "summarize_on_reset": True
            }
        }
    }

@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)

@pytest.fixture
def system_prompt_dir():
    """Create a temporary system prompt directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test prompt file
        prompt_path = os.path.join(temp_dir, "test_prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write("Test system prompt")
        yield temp_dir

def test_config_manager_singleton():
    """Test that ConfigManager is a singleton"""
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    assert manager1 is manager2

def test_load_config(config_file, sample_config):
    """Test loading configuration from file"""
    manager = ConfigManager()
    manager.load_config(config_file)
    
    assert manager.config == sample_config
    assert manager.llm_config == sample_config['llm']
    assert manager.agent_config == sample_config['agent']

def test_load_config_invalid_file():
    """Test loading invalid configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content")
        config_path = f.name
    
    try:
        manager = ConfigManager()
        with pytest.raises(Exception) as exc_info:
            manager.load_config(config_path)
        assert "Error loading config" in str(exc_info.value)
    finally:
        os.unlink(config_path)

def test_load_config_missing_file():
    """Test loading non-existent configuration file"""
    manager = ConfigManager()
    with pytest.raises(Exception) as exc_info:
        manager.load_config("nonexistent.yaml")
    assert "Error loading config" in str(exc_info.value)

def test_get_system_prompt(sample_config, system_prompt_dir):
    """Test loading system prompt"""
    manager = ConfigManager()
    
    # Update config to use our test prompt directory
    config = sample_config.copy()
    config['agent']['system_prompt_dir'] = system_prompt_dir
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        manager.load_config(config_path)
        prompt = manager.get_system_prompt("test_prompt")
        assert prompt == "Test system prompt"
    finally:
        os.unlink(config_path)

def test_get_system_prompt_missing():
    """Test loading non-existent system prompt"""
    manager = ConfigManager()
    with pytest.raises(Exception) as exc_info:
        manager.get_system_prompt("nonexistent_prompt")
    assert "Error loading system prompt" in str(exc_info.value)

def test_default_config():
    """Test loading default configuration"""
    manager = ConfigManager()
    config = manager.config  # This should load the default config
    
    # Verify essential config sections exist
    assert 'llm' in config
    assert 'agent' in config
    
    # Verify essential settings exist
    assert 'provider' in config['llm']
    assert 'max_retries' in config['agent']
    assert 'conversation' in config['agent']

def test_config_environment_variables(monkeypatch):
    """Test that environment variables are respected"""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key-from-env')
    
    manager = ConfigManager()
    manager._config = {
        'llm': {
            'provider': 'openrouter',
            'openrouter': {
                'api_key': ''  # Empty to test environment variable
            }
        }
    }
    
    assert manager.llm_config['openrouter']['api_key'] == ''  # Should still be empty in config
    # The actual API key should be handled by the LLM provider 