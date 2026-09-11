@echo off
echo ============================================================
echo  CycloNex — GitHub Push Helper
echo  GitHub account: ssubham07
echo ============================================================
echo.
echo STEP 1: Make sure repo "cyclonex" exists on GitHub.
echo         Go to: https://github.com/new
echo         Name: cyclonex
echo         Keep it PUBLIC, NO README, NO .gitignore
echo         Click "Create repository"
echo.
echo STEP 2: Run the push command below.
echo         When prompted:
echo           Username: ssubham07
echo           Password: [your GitHub Personal Access Token]
echo.
echo         To create a PAT:
echo         https://github.com/settings/tokens/new
echo         Select scope: repo (full control)
echo.
echo ============================================================
echo.

set PATH=%PATH%;%USERPROFILE%\scoop\shims;%USERPROFILE%\scoop\apps\git\current\cmd

cd /d "C:\Users\pradh\.gemini\antigravity\scratch\cyclonex"
git push -u origin main

echo.
echo ============================================================
echo  Done! Visit: https://github.com/ssubham07/cyclonex
echo ============================================================
pause
