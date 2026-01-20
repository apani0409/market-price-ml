#!/bin/bash
set -e

echo "Installing Python dependencies..."

# Intenta diferentes métodos para instalar
if pip3 install --break-system-packages -r requirements.txt 2>/dev/null; then
    echo "✓ Instaladas con pip3 --break-system-packages"
elif uv pip install -r requirements.txt 2>/dev/null; then
    echo "✓ Instaladas con uv pip"
elif pip install -r requirements.txt 2>/dev/null; then
    echo "✓ Instaladas con pip"
else
    echo "✗ Error instalando dependencias"
    exit 1
fi

echo "Build completed successfully"
