@echo off
setlocal enabledelayedexpansion

title Titans MAC Training
cd /d "%~dp0"

:: ===== Config =====
set LOGFILE=train_output.log
set CHECKPOINT_DIR=checkpoints

:: ===== Checks =====
echo.
echo =============================================
echo   Titans MAC - Pure Implementation Training
echo =============================================
echo.
echo Checking environment...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Make sure Python is in your PATH.
    pause
    exit /b 1
)

python -c "import torch; print('  PyTorch:', torch.__version__); print('  Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch not installed. Run: pip install torch
    pause
    exit /b 1
)

python -c "import titans_pytorch; print('  titans-pytorch:', titans_pytorch.__version__)" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] titans-pytorch not installed. Run: pip install titans-pytorch
    pause
    exit /b 1
)

python -c "import datasets; print('  datasets:', datasets.__version__)" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] datasets not installed. Run: pip install datasets
    pause
    exit /b 1
)

if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

:: ===== Show config =====
echo.
echo ---------------------------------------------
echo  Model: Titans MAC
echo  Dim: 384, Depth: 6, Heads: 6
echo  Seg len: 128, Seq len: 512
echo  Persist mem: 8, Long-term mem: 16
echo  Parameters: ~49M
echo.
echo  Dataset: FineWeb sample-10BT (50k docs)
echo  Steps: 50,000
echo  Save every: 2,000 steps
echo.
echo  Output: %CHECKPOINT_DIR%\
echo  Log: %LOGFILE%
echo ---------------------------------------------
echo.

:: ===== Ask to resume =====
set RESUME_FLAG=
dir /b "%CHECKPOINT_DIR%\*.pt" >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] Existing checkpoints found:
    dir /b /o-d "%CHECKPOINT_DIR%\*.pt" 2>nul
    echo.
    set /p RESUME_FLAG="Resume from latest checkpoint? (Y/n): "
    if /i "!RESUME_FLAG!"=="" set RESUME_FLAG=Y
)

echo.
echo =============================================
echo  Starting at: %date% %time%
echo =============================================
echo.

:: ===== Run training =====
python -u "%~dp0train_titans.py" 2>&1 | tee "%LOGFILE%"
set EXIT_CODE=%errorlevel%

echo.
echo =============================================
if %EXIT_CODE% equ 0 (
    echo  Training finished! Checkpoints in %CHECKPOINT_DIR%\
    echo  Run generate.py to test the model.
) else (
    echo  Training stopped (exit code: %EXIT_CODE%^)
    echo  Check %LOGFILE% for details.
)
echo =============================================
echo.

:: ===== Show final checkpoint =====
dir /b /o-d "%CHECKPOINT_DIR%\*.pt" 2>nul | find /v "" >nul
if !errorlevel! equ 0 (
    echo Latest checkpoint:
    for /f %%f in ('dir /b /o-d "%CHECKPOINT_DIR%\*.pt"') do (
        for %%s in ("%CHECKPOINT_DIR%\%%f") do (
            set size=%%~zs
            if !size! gtr 1048576 (
                set /a "mb=!size! / 1048576"
                echo   %%f (!mb! MB^)
            ) else (
                set /a "kb=!size! / 1024"
                echo   %%f (!kb! KB^)
            )
        )
        goto :break
    )
)
:break

echo.
pause
