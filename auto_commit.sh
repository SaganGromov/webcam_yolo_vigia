#!/bin/bash

# Directories to monitor
WATCH_DIRS=("motion_frames_detected" "person_frames_detected" "logs/motion_logs" "logs/person_logs")

# Git repository location
REPO_PATH="$HOME/vigiar"

# Navigate to the repository
cd "$REPO_PATH" || { echo "Error: Could not change to directory $REPO_PATH"; exit 1; }

# Check for changes
if git diff --quiet && git diff --cached --quiet && git ls-files --others --exclude-standard --quiet; then
  echo "No changes detected."
else
  # Stage changes
  git add .
  
  # Commit changes with a timestamp
  COMMIT_MESSAGE="Auto-commit on $(date '+%Y-%m-%d %H:%M:%S')"
  git commit -m "$COMMIT_MESSAGE"
  
  # Push to the repository
  git push origin main
  echo "Changes committed and pushed to GitHub."
fi
