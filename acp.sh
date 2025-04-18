if [ $# -lt 1 ]; then
    echo "Usage: $0 '<commit message>'"
    exit 1
fi

MESSAGE="$1"

git add .
git commit -m "$MESSAGE"
git push
