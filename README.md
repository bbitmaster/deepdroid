# DeepDroid

DeepDroid is a minimal framework for building LLM-powered agents that can iteratively solve tasks. It's designed to be simple, extensible, and run in a containerized environment for safety.

## What is DeepDroid?

DeepDroid represents a new paradigm in AI development - the concept of autonomous AI agents. The name "DeepDroid" is a playful nod to both the era when everything in machine learning was prefixed with "deep" and the word "droid" which evokes the idea of an autonomous AI agent. While the terminology in this space is still evolving, the core concept is powerful: giving AI systems a degree of autonomy through well-defined contracts and capabilities.

At its heart, DeepDroid allows you to create AI agents that can:
- Make decisions autonomously within defined boundaries
- Execute real actions through code and system commands
- Maintain memory and context across sessions
- Iteratively solve complex tasks
- Interact with and improve codebases

## Why DeepDroid?

The field of AI is rapidly evolving, and we're entering an era where Large Language Models (LLMs) are becoming capable enough to handle complex, multi-step tasks with remarkable reliability. This creates an opportunity for a new kind of software architecture - one where AI agents can be given:

1. **A Mission**: A high-level objective or task to accomplish
2. **A Contract**: A well-defined set of capabilities and constraints
3. **Tools**: The ability to execute code, run commands, and interact with systems
4. **Memory**: Persistent storage to maintain context and learning across sessions

DeepDroid provides a minimal framework for building these kinds of agents. As LLM capabilities improve and inference costs decrease, frameworks like this will become increasingly powerful, potentially enabling:
- Automated codebase improvements and maintenance
- Complex system administration tasks
- Autonomous research and experimentation
- Collaborative problem-solving between multiple agents

The framework is intentionally minimal, providing only the core components needed:
- A system prompt that defines the agent's behavior and capabilities
- A parser that executes the agent's decisions
- A task runner that coordinates the interaction loop
- Persistent memory for maintaining context

## Author

**Ben Goodrich (with AI assistance)**  
Email: bbitmaster@gmail.com  
LinkedIn: [Ben Goodrich](https://www.linkedin.com/in/ben-goodrich-5740a7b1/)

## Getting Started

1. **Get an API Key**:
   - Go to [OpenRouter](https://openrouter.ai/)
   - Sign up and get credits
   - Create an API key in your dashboard
   - Export it as an environment variable:
     ```bash
     export OPENROUTER_API_KEY=your_key_here
     ```

2. **Setup**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/deepdroid
   cd deepdroid

   # Create directories for code and persistent memory
   mkdir -p workspace data

   # Build the container
   docker build -t deepdroid .
   ```

3. **Quick Test**:
   ```bash
   # Test the LLM connection
   docker run -it --rm --network host \
       -v $(pwd)/data:/app/data \
       -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
       deepdroid test "Hello, world!"
   ```

## Usage

### Running the Codebase Improver

1. **Copy code to analyze**:
   ```bash
   # Copy your code into the workspace directory
   cp -r /path/to/your/code/* workspace/
   ```

2. **Run the improver**:
   ```bash
   docker run -it \
       --network host \
       -v $(pwd)/workspace:/app/workspace \
       -v $(pwd)/data:/app/data \
       -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
       deepdroid improve
   ```

   Or use the provided script (recommended):
   ```bash
   ./scripts/run.sh improve
   ```

### Advanced Options

```bash
# Run with custom message
docker run -it \
    --network host \
    -v $(pwd)/workspace:/app/workspace \
    -v $(pwd)/data:/app/data \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid improve --message "Analyze the code for security issues"

# Run with maximum iterations
docker run -it \
    --network host \
    -v $(pwd)/workspace:/app/workspace \
    -v $(pwd)/data:/app/data \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid improve --max-iterations 5

# Test with different models
docker run -it \
    --network host \
    -v $(pwd)/data:/app/data \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid test --model anthropic/claude-3-opus "Hello, world!"
```

## Configuration

The default configuration uses OpenRouter with Claude 3 Sonnet. Available models include:
- `anthropic/claude-3-opus` (most capable)
- `anthropic/claude-3-sonnet` (good balance)
- `openai/gpt-4-turbo`

To use a custom configuration:
```bash
# Copy and edit the default config
cp deepdroid/config/agent_config.yaml ./my_config.yaml

# Run with custom config
docker run -it \
    --network host \
    -v $(pwd)/workspace:/app/workspace \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/my_config.yaml:/app/config.yaml \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid improve --config /app/config.yaml
```

## Understanding Persistent Memory

DeepDroid maintains memory between runs to provide context and continuity:
- Memory is stored in `./data/agent_memory.txt`
- Helps maintain context across sessions
- Automatically persists between container runs
- To reset memory: `rm data/agent_memory.txt`

## Troubleshooting

Common issues:
1. **API Key Not Found**: Make sure `OPENROUTER_API_KEY` is set in your environment
2. **Model Not Available**: Check OpenRouter dashboard for available models and credits
3. **Rate Limits**: OpenRouter has rate limits based on your tier
4. **Workspace Permissions**: Ensure the mounted workspace is writable
5. **Network Issues**: Make sure to use `--network host` flag for internet access
6. **Docker Mount Issues**: Verify you're in the correct directory when mounting volumes

## Security Considerations

DeepDroid can execute commands and modify files in its workspace. To minimize risks:
1. Only mount the specific directories needed (`workspace` and `data`)
2. Review the system prompts to understand allowed actions
3. Back up important files before modification
4. Use the test command to verify behavior
5. Run in container with limited permissions

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"

# Run tests
pytest

# Run with debug logging
deepdroid --debug improve
```

## License

MIT License - See LICENSE file for details