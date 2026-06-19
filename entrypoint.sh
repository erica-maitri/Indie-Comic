#!/bin/bash
set -e

echo "================================================================="
echo "🚀 INITIALIZING ALL-IN-ONE COMIC GENERATOR CONTAINER"
echo "================================================================="

# Wait for Ollama service to start up and be healthy
OLLAMA_HOST=${OLLAMA_URL:-"http://ollama:11434"}
echo "Checking connection to Ollama at: $OLLAMA_HOST..."

until curl -s "$OLLAMA_HOST/api/tags" > /dev/null; do
  echo "Waiting for Ollama service to be ready... (retrying in 3 seconds)"
  sleep 3
done

echo "✅ Connected to Ollama successfully!"

# Pull LLaMA 3.2 model if it doesn't exist
echo "Ensuring llama3.2 model is downloaded on Ollama..."
curl -s -X POST "$OLLAMA_HOST/api/pull" -d '{"name": "llama3.2"}' > /dev/null
echo "✅ llama3.2 model is ready for generation!"

echo "================================================================="
echo "🎉 ALL-IN-ONE PIPELINE ENVIRONMENT INITIALIZED SUCCESSFULLY!"
echo "================================================================="
echo "To run the generator inside this container, execute:"
echo "  docker compose exec comic-generator python indie_comic_pipeline/integrated_pipeline.py"
echo "================================================================="

# Keep container alive
exec tail -f /dev/null
