#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Check if a commit message was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 '<commit message>'"
    exit 1
fi

MESSAGE="$1"

# Run pre-commit checks on all files
echo "Running pre-commit checks..."
pre-commit run --all-files
PRE_COMMIT_EXIT_CODE=$?

# Only proceed if pre-commit checks pass
if [ $PRE_COMMIT_EXIT_CODE -ne 0 ]; then
    echo "❌ Pre-commit checks failed. Please fix the issues and try again."
    exit 1
else
    echo "✅ Pre-commit checks passed."
fi

# Add, commit, and push
echo "Adding changes..."
git add .

echo "Committing with message: \"$MESSAGE\""
git commit -m "$MESSAGE"

echo "Pushing to remote repository..."
git push origin main

echo "✅ Changes successfully added, committed, and pushed!"
