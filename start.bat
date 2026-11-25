@echo off
rem start.bat - double-click to run any part of the BIM Tutor.
rem It goes to this folder and activates .venv for you, so you never
rem have to type cd / activate yourself.

cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Could not find .venv in %cd%
    echo Make sure start.bat is inside the sign-tutor folder.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:menu
echo.
echo ===== BIM Tutor =====
echo   1  Record samples   (collect_data.py)
echo   2  Train the model  (train_model.py)
echo   3  Live recognition (recognizer.py)
echo   4  Tutor game       (tutor.py)
echo   --- moving signs ---
echo   5  Record clips     (collect_dynamic.py)
echo   6  Train            (train_dynamic.py)
echo   7  Live test        (dynamic_live.py)
echo   Q  Quit
echo.
set "choice="
set /p "choice=Choose 1-7 or Q: "

if "%choice%"=="1" python collect_data.py
if "%choice%"=="2" python train_model.py
if "%choice%"=="3" python recognizer.py
if "%choice%"=="4" python tutor.py
if "%choice%"=="5" python collect_dynamic.py
if "%choice%"=="6" python train_dynamic.py
if "%choice%"=="7" python dynamic_live.py
if /i "%choice%"=="q" exit /b 0
goto menu
