# gpt-engineer

The OG code generation experimentation platform!

gpt-engineer lets you:
- Specify software in natural language
- Sit back and watch as an AI writes and executes the code
- Ask the AI to implement improvements

## Getting Started

### Install gpt-engineer

For stable release:
- python -m pip install gpt-engineer

For development:
- git clone https://github.com/gpt-engineer-org/gpt-engineer.git
- cd gpt-engineer
- poetry install
- poetry shell to activate the virtual environment

We actively support Python 3.10 - 3.12. The last version to support Python 3.8 - 3.9 was 0.2.6.

### Setup API key

Choose one of:
- Export env variable: export OPENAI_API_KEY=[your api key]
- .env file: Create a copy of .env.template named .env and add your OPENAI_API_KEY in .env
- Custom model: See docs, supports local model, azure, etc.

### Create new code (default usage)

Create an empty folder for your project anywhere on your computer. Create a file called prompt (no extension) inside your new folder and fill it with instructions. Run gpte <project_dir> with a relative path to your folder.

### Improve existing code

Locate a folder with code which you want to improve anywhere on your computer. Create a file called prompt (no extension) inside your new folder and fill it with instructions for how you want to improve the code. Run gpte <project_dir> -i with a relative path to your folder.

### Benchmark custom agents

gpt-engineer installs the binary 'bench', which gives you a simple interface for benchmarking your own agent implementations against popular public datasets. The easiest way to get started with benchmarking is by checking out the template repo, which contains detailed instructions and an agent template. Currently supported benchmarks: APPS and MBPP.

## Features

### Pre Prompts

You can specify the "identity" of the AI agent by overriding the preprompts folder with your own version of the preprompts. You can do so via the --use-custom-preprompts argument. Editing the preprompts is how you make the agent remember things between projects.

### Vision

By default, gpt-engineer expects text input via a prompt file. It can also accept image inputs for vision-capable models. This can be useful for adding UX or architecture diagrams as additional context for GPT Engineer. You can do this by specifying an image directory with the --image_directory flag and setting a vision-capable model in the second CLI argument.

### Open source, local and alternative models

By default, gpt-engineer supports OpenAI Models via the OpenAI API or Azure OpenAI API, as well as Anthropic models. With a little extra setup, you can also run with open source models like WizardCoder.

## Mission

The gpt-engineer community mission is to maintain tools that coding agent builders can use and facilitate collaboration in the open source community. If you are interested in contributing to this, we are interested in having you. If you want to see our broader ambitions, check out the roadmap, and join discord to learn how you can contribute to it.

gpt-engineer is governed by a board of long-term contributors. If you contribute routinely and have an interest in shaping the future of gpt-engineer, you will be considered for the board.