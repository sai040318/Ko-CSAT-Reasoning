#!/bin/bash

set -e

context_length=${1:-4096}
echo "Starting Ollama server with context length: $context_length"
OLLAMA_CONTEXT_LENGTH=$context_length ollama serve
