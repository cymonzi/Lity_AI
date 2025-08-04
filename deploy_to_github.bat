@echo off
echo 🚀 Preparing Lity AI for GitHub Pages deployment...
echo.

echo 📋 Step 1: Switching to production chat logic...
copy /Y "src\chatLogic.production.js" "src\chatLogic.js"
if errorlevel 1 (
    echo ❌ Failed to copy production chat logic
    pause
    exit /b 1
)
echo ✅ Production chat logic activated

echo.
echo 📦 Step 2: Building the project...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed
    pause
    exit /b 1
)
echo ✅ Build completed successfully

echo.
echo 🌐 Step 3: Deploying to GitHub Pages...
call npm run deploy
if errorlevel 1 (
    echo ❌ Deployment failed
    pause
    exit /b 1
)

echo.
echo 🎉 Deployment completed successfully!
echo 🔗 Your site will be available at: https://cymonzi.github.io/Lity_AI/
echo.
echo 📝 Note: It may take a few minutes for changes to appear on GitHub Pages
echo.
pause
