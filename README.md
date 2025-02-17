# deepdroid
deepdroid is a very simple agent framework that can use LLMs to iteratively solve tasks.

Note that I am calling these "agents" for lack of a better term. I believe we are still trying to work out good abstractions and design patterns for this space.

I am defining an agent very broadly as a system that uses an LLM to decide the control flow of an application.

More specifically, an agent is a system that has access to an LLM, and can use the LLM to make decisions. In this sense it has some amount of autonomy.

The framework in deepdroid is meant to be very minimal, and provide only the core functionality that is needed to build agents.

In this way we have several main components:
1. A system prompt for the LLM. This prompt is used both to guide the behavior of the agent and to provide a language for the agent to use.
2. A parser that parses the LLMs responses and executes the actions that the agent gives.
3. A task runner that is responsible for running the agent and the parser in a loop until the task is complete.

## System Prompt

The system prompt is a string that is used to guide the behavior of the agent. It is used both to guide the behavior of the agent and to provide a language for the agent to use.

## Parser

The parser is responsible for parsing the LLMs responses and executing the actions that the agent gives.
Note that the parser is responsible for a sequence of back and forths between the LLM and executes a series of actions providing feedback to the LLM before being instructed to execute the next action.

After some number of actions the parser will reset the context window of the LLM. However, the parser itself can maintain its own state and even rebuilt the system prompt if needed.

It's probably wise to give the agent a set of tools to store and retrieve information from a memory that will be globally provided even if the context window is reset.

## Task Runner

This is the main loop that will run the agent and the parser in a loop until the task is complete or some criteria is met.

## Running environment

Currently we have parsers that can run in a local environment and perform bash commands and create files. This is possible dangerous, as it can give an LLM free reign to do whatever it wants.

For now we will use a docker container to run the parsers in a somewhat controlled environment, but this is by no means meant to be secure.