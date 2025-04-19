#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Check if a commit message was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 '<commit message>' [--no-verify]"
    exit 1
fi

MESSAGE="$1"
BYPASS_PRECOMMIT=false

# Check for the --no-verify flag
if [ $# -gt 1 ] && [ "$2" = "--no-verify" ]; then
    BYPASS_PRECOMMIT=true
    echo "⚠️ Pre-commit checks will be bypassed as requested."
fi

# Run pre-commit checks on all files if not bypassed
if [ "$BYPASS_PRECOMMIT" = false ]; then
    echo "Running pre-commit checks..."
    pre-commit run --all-files
    PRE_COMMIT_EXIT_CODE=$?

    # Only proceed if pre-commit checks pass
    if [ $PRE_COMMIT_EXIT_CODE -ne 0 ]; then
        echo "❌ Pre-commit checks failed. Please fix the issues or use --no-verify to bypass."
        exit 1
    else
        echo "✅ Pre-commit checks passed."
    fi
fi

# Add, commit, and push
echo "Adding changes..."
git add .

if [ "$BYPASS_PRECOMMIT" = true ]; then
    echo "Committing with message: \"$MESSAGE\" (bypassing pre-commit hooks)"
    git commit -m "$MESSAGE" --no-verify
else
    echo "Committing with message: \"$MESSAGE\""
    git commit -m "$MESSAGE"
fi

echo "Pushing to remote repository..."
git push origin main

echo "✅ Changes successfully added, committed, and pushed!" 