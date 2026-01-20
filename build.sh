#!/bin/bash
set -e

echo "Installing dependencies with uv..."
uv pip install -r requirements.txt

echo "Build completed successfully!"
