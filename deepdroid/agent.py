"""
Core agent implementation
"""

class Agent:
    def __init__(self, parser_class, config_path: str = None):
        self.parser_class = parser_class
        self.config_path = config_path
        self.parser = None

    async def process_message(self, message: str, system_prompt_name: str):
        # In production this would integrate with an LLM to process the message
        # For now, we raise NotImplementedError; during tests, this method will be mocked.
        raise NotImplementedError("process_message is not implemented")

    async def __aenter__(self):
        self.parser = self.parser_class()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        # Perform any necessary cleanup here
        pass 