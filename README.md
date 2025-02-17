# DeepDroid

DeepDroid is a minimal framework for building LLM-powered agents that can iteratively solve tasks. It's designed to be simple, extensible, and run in a containerized environment for safety.

## Author

**Ben Goodrich**  
Email: bbitmaster@gmail.com  
LinkedIn: [Ben Goodrich](https://www.linkedin.com/in/ben-goodrich-5740a7b1/)

## Getting Started

1. **Get an API Key**:
   - Go to [OpenRouter](https://openrouter.ai/)
   - Sign up and get credits (they offer free credits to start)
   - Create an API key in your dashboard
   - Export it as an environment variable:
     ```bash
     export OPENROUTER_API_KEY=your_key_here
     ```

2. **Choose Your Setup**:

   **Option A: Local Installation** (Not recommended for production)
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/deepdroid
   cd deepdroid

   # Install the package
   pip install -e .
   ```

   **Option B: Docker Setup** (Recommended)
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/deepdroid
   cd deepdroid

   # Build the container
   docker build -t deepdroid .

   # Create a workspace directory for the agent
   mkdir workspace
   ```

3. **Quick Test**:

   **Local:**
   ```bash
   deepdroid test "Hello, world!"
   ```

   **Docker:**
   ```bash
   docker run -it --rm \
       -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
       deepdroid test "Hello, world!"
   ```

## Usage

DeepDroid provides a command-line interface with two main commands. These can be run either locally or in Docker.

### Testing LLM Connection

Test your LLM provider configuration:

```bash
# Local usage
deepdroid test "What is the capital of France?"
deepdroid test -v "Hello, world!"
deepdroid test --model anthropic/claude-3-opus "Hello, world!"

# Docker usage
docker run -it --rm \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid test "What is the capital of France?"
```

### Running the Codebase Improver

The main agent that analyzes and improves codebases:

**Local Usage** (operates on current directory):
```bash
# Basic usage
deepdroid improve

# With options
deepdroid improve --config custom_config.yaml \
                 --max-iterations 5 \
                 --message "Analyze the code for security issues"
```

**Docker Usage** (operates on mounted workspace):
```bash
# First, copy or move the code you want to analyze into the workspace
cp -r /path/to/your/code deepdroid/workspace/

# Run the improver in the container
docker run -it \
    -v $(pwd)/workspace:/app/workspace \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    deepdroid improve

# Or use the provided script (recommended)
./scripts/run.sh improve
```

## Docker vs Local Usage

### Docker (Recommended)
- **Pros:**
  - Isolated environment
  - Can't accidentally modify system files
  - Consistent environment across systems
  - Required dependencies pre-installed
- **Cons:**
  - Need to copy code into workspace directory
  - Extra setup step
- **When to use:**
  - Production environments
  - When modifying important code
  - When running on shared systems

### Local Installation
- **Pros:**
  - Works directly on current directory
  - No need to copy files
  - Simpler setup
- **Cons:**
  - Less secure
  - Can modify system files
  - Dependencies might conflict
- **When to use:**
  - Development/testing
  - Quick experiments
  - Trusted environments

## Configuration

The default configuration uses OpenRouter with Claude 3 Sonnet. You can customize this by creating an `agent_config.yaml` file:

```yaml
llm:
  provider: openrouter
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: ""  # Set via OPENROUTER_API_KEY env var
    default_model: anthropic/claude-3-sonnet  # or claude-3-opus, gpt-4-turbo
    timeout: 30

agent:
  max_retries: 3
  retry_delay: 1
  memory_file: "agent_memory.txt"
  system_prompt_dir: "system_prompts"
  log_level: "INFO"
  conversation:
    max_turns: 10
    max_chars: 32000
    include_tool_outputs: true
    summarize_on_reset: true
```

Available OpenRouter models include:
- `anthropic/claude-3-opus` (most capable)
- `anthropic/claude-3-sonnet` (good balance)
- `openai/gpt-4-turbo` 
- See [OpenRouter docs](https://openrouter.ai/docs) for more

## Docker Workspace Setup

The Docker container expects code to be in the `/app/workspace` directory. Here's how to set it up:

1. **Create workspace directory:**
   ```bash
   mkdir -p deepdroid/workspace
   ```

2. **Add code to analyze:**
   ```bash
   # Option 1: Copy existing code
   cp -r /path/to/your/code deepdroid/workspace/

   # Option 2: Clone a repository
   cd deepdroid/workspace
   git clone https://github.com/user/repo .
   ```

3. **Run the container:**
   ```bash
   cd deepdroid  # Make sure you're in the deepdroid root
   docker run -it \
       -v $(pwd)/workspace:/app/workspace \
       -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
       deepdroid improve
   ```

## Troubleshooting

Common issues:
1. **API Key Not Found**: Make sure `OPENROUTER_API_KEY` is set in your environment
2. **Model Not Available**: Check OpenRouter dashboard for available models and credits
3. **Rate Limits**: OpenRouter has rate limits based on your tier
4. **Workspace Permissions**: When using Docker, ensure the mounted workspace is writable
5. **Docker Mount Issues**: Ensure you're mounting the workspace from the correct directory

## Security Considerations

DeepDroid can execute commands and modify files in its workspace. To minimize risks:

1. Always run it in a container with limited permissions
2. Mount only the specific directories it needs access to
3. Review the system prompts to understand what actions are allowed
4. Use the test command to verify behavior before running the improver
5. Back up any important files before letting the agent modify them

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"

# Run tests
pytest

# Run with debug logging
deepdroid --debug improve
```

## Contributing

Contributions are welcome! Some areas that could use improvement:

1. Additional parsers for different tasks
2. Support for more LLM providers
3. Enhanced safety mechanisms
4. Better test coverage
5. Documentation improvements
6. Simplified Docker workflow

## License

MIT License - See LICENSE file for details