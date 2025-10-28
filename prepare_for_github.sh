#!/bin/bash

echo "🚀 Preparing Credit Risk Demo for GitHub"
echo "========================================"

# Remove temporary files
echo "🧹 Cleaning up temporary files..."
rm -f test_imports.py
rm -f setup_python_alias.sh
rm -f readme.md  # Old readme (we have README.md now)

# Create necessary directories if they don't exist
echo "📁 Ensuring directories exist..."
mkdir -p models
mkdir -p artifacts
mkdir -p eda_plots
mkdir -p validation_plots
mkdir -p data
mkdir -p notebooks

# Check if .git exists
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
else
    echo "✅ Git repository already initialized"
fi

# Check git status
echo ""
echo "📊 Current git status:"
git status

echo ""
echo "✅ Preparation complete!"
echo ""
echo "📝 Next steps:"
echo "1. Review files to commit: git status"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Initial commit: Credit Risk Model Demo'"
echo "4. Create GitHub repo and add remote: git remote add origin <your-repo-url>"
echo "5. Push: git push -u origin main"
echo ""
echo "💡 Optional: Remove old readme.md if it still exists"
echo "   rm readme.md"

