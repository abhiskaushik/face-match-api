#!/bin/bash
set -e

echo "==================================="
echo "  Face Match API — Deploy Script"
echo "==================================="
echo ""

# ---- Step 1: Check prerequisites ----
echo "Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "❌ git is not installed. Install it: https://git-scm.com"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) not found."
    echo "   Install it: https://cli.github.com"
    echo "   Then run: gh auth login"
    echo ""
    echo "   Alternatively, create the repo manually on github.com"
    echo "   and push with:"
    echo "     git remote add origin https://github.com/YOUR_USER/face-match-api.git"
    echo "     git push -u origin main"
    echo ""
    SKIP_GH=true
else
    SKIP_GH=false
fi

# ---- Step 2: Initialize git repo ----
if [ ! -d ".git" ]; then
    echo "Initializing git repo..."
    git init
    git add -A
    git commit -m "Initial commit: Face Match API with InsightFace"
fi

# ---- Step 3: Create GitHub repo & push ----
if [ "$SKIP_GH" = false ]; then
    echo ""
    echo "Creating GitHub repository..."
    gh repo create face-match-api --public --source=. --push
    echo "✅ Repository pushed to GitHub"
else
    echo ""
    echo "Skipping GitHub repo creation (no gh CLI)."
    echo "Create the repo manually and push, then continue."
fi

# ---- Step 4: Railway deployment ----
echo ""
echo "==================================="
echo "  Deploy to Railway"
echo "==================================="
echo ""

if command -v railway &> /dev/null; then
    echo "Railway CLI found. Deploying..."
    railway login
    railway init
    railway up
    echo ""
    echo "✅ Deployed! Run 'railway open' to see your app."
else
    echo "To deploy on Railway:"
    echo ""
    echo "  Option A — Railway CLI:"
    echo "    npm install -g @railway/cli"
    echo "    railway login"
    echo "    railway init"
    echo "    railway up"
    echo ""
    echo "  Option B — Railway Web UI:"
    echo "    1. Go to https://railway.app/new"
    echo "    2. Sign up with GitHub"
    echo "    3. Click 'Deploy from GitHub Repo'"
    echo "    4. Select 'face-match-api'"
    echo "    5. Railway auto-detects the Dockerfile and deploys"
    echo "    6. Go to Settings → Networking → Generate Domain"
    echo ""
    echo "  The first deploy takes ~5 min (downloads the 326MB model)."
    echo "  After that, your app will be live at https://your-app.up.railway.app"
fi

echo ""
echo "Done! 🎉"
