@echo off
rem ===========================================================================
rem  run_Python.bat
rem ---------------------------------------------------------------------------
rem  Numerical experiments for the article
rem
rem    D. Gulua, J. Rogava, Z. Vashakidze,
rem    "Perturbation algorithm and iterative realization of a four-layer
rem     semi-discrete scheme for an evolutionary problem with a variable
rem     operator", Proceedings of I. Vekua Institute of Applied Mathematics,
rem     Vol. 76, 2026 (preprint: https://arxiv.org/abs/TBA).
rem
rem  Repository: https://github.com/zv1991/four-layer-perturbation-iterative-scheme
rem  The script must stay in the same folder as the Python files.
rem
rem  What this script does:
rem    1. finds a Python 3 interpreter (the "py" launcher or "python") and
rem       checks its version (Python 3.8 or newer is required);
rem    2. checks the Python packages listed in requirements.txt; if some
rem       of them are missing or outdated, it lists them and asks whether to
rem       install them with pip; after the installation it checks them again;
rem    3. runs the self-test of the implementation (self_test.py);
rem    4. deletes the CSV files of the previous run from the folder "data"
rem       (the list of files is obtained from run_experiments.py);
rem    5. runs run_experiments.py, which writes new CSV files to "data"
rem       (the data of the tables and figures of Section 7 of the article);
rem    6. checks that all expected CSV files have been created;
rem    7. removes the Python cache folder (__pycache__), if any.
rem
rem  Usage:   double-click the file, or type   run_Python.bat   in a console.
rem           run_Python.bat /nopause   -  do not wait for a key at the end
rem                                       (e.g. when called from another script).
rem
rem  The question about the installation of missing packages is answered
rem  "N" (no installation) automatically if there is no answer within
rem  60 seconds.
rem ===========================================================================

setlocal EnableExtensions DisableDelayedExpansion
title Numerical experiments - Gulua, Rogava, Vashakidze

rem --- Work in the folder of this script, wherever it is called from --------
cd /d "%~dp0"

rem --- Settings ---------------------------------------------------------------
rem  Python scripts (relative to this folder)
set "SCRIPT=run_experiments.py"
set "CHECKER=check_requirements.py"
set "SELFTEST=self_test.py"
set "REQFILE=requirements.txt"
rem  Output folder of the CSV files (relative to this folder)
set "DATADIR=data"
rem  Seconds to wait for an answer to the installation question
set "ANSWER_TIMEOUT=60"
rem  Total number of steps shown in the progress bar
set "STEPS=7"

rem --- Command-line option /nopause ----------------------------------------
set "NOPAUSE="
if /i "%~1"=="/nopause" set "NOPAUSE=1"

set "T_START=%TIME%"
echo.
echo  ===========================================================================
echo   Numerical experiments: perturbation algorithm for a four-layer scheme
echo  ===========================================================================

rem --- The Python scripts must be present -----------------------------------
for %%f in ("%SCRIPT%" "%CHECKER%" "%SELFTEST%" "%REQFILE%") do (
    if not exist "%%~f" (
        echo.
        echo  [ERROR] Missing file: %%~f
        goto :fail
    )
)

rem ===========================================================================
rem  Step 1: find a Python 3 interpreter (version 3.8 or newer)
rem ===========================================================================
call :progress 1 "Searching for a Python 3 interpreter"
set "PY="
rem  The Python launcher "py" is preferred; then "python" from PATH.
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo.
    echo  [ERROR] Python 3 was not found.
    echo          Install Python 3 from https://www.python.org/downloads/
    echo          and enable the option "Add python.exe to PATH".
    goto :fail
)
for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%v"
echo          Using: %PY%  (Python %PYVER%)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python 3.8 or newer is required. Please update Python.
    goto :fail
)

rem ===========================================================================
rem  Step 2: check the required packages and offer to install missing ones
rem ===========================================================================
call :progress 2 "Checking the required Python packages"
%PY% -B "%CHECKER%"
set "CHECK_RESULT=%ERRORLEVEL%"
if "%CHECK_RESULT%"=="0" (
    echo          All required packages are installed.
    goto :self_test
)
if not "%CHECK_RESULT%"=="1" (
    echo  [ERROR] The requirements could not be checked ^(see above^).
    goto :fail
)
echo.
echo  [WARNING] The package^(s^) marked above as MISSING or OUTDATED are needed
echo            for the numerical experiments.
echo            They can be installed now with pip ^(an Internet connection is
echo            required^). If there is no answer within %ANSWER_TIMEOUT% s, nothing is
echo            installed.
echo.
choice /C YN /T %ANSWER_TIMEOUT% /D N /M "          Install the missing package(s) now"
if errorlevel 2 (
    echo.
    echo  [ERROR] The experiments cannot be run without the required package^(s^).
    echo          You can install them later with:
    echo              %PY% -m pip install -r %REQFILE%
    goto :fail
)
echo.
echo          Installing the missing package^(s^) ...
%PY% -B "%CHECKER%" --install
if errorlevel 1 (
    echo.
    echo  [ERROR] The installation failed. See the pip messages above.
    echo          Check the Internet connection, or install the packages manually:
    echo              %PY% -m pip install -r %REQFILE%
    goto :fail
)
rem  Check again in a new Python process (a new process also sees packages
rem  installed into a user folder that did not exist before)
echo.
echo          Checking the packages again:
%PY% -B "%CHECKER%"
if errorlevel 1 (
    echo  [ERROR] The packages still cannot be imported after the installation.
    echo          Close this window and run run_Python.bat once more.
    goto :fail
)
echo          The missing package^(s^) have been installed successfully.

rem ===========================================================================
rem  Step 3: self-test of the implementation
rem ===========================================================================
:self_test
call :progress 3 "Self-test of the implementation"
%PY% -B "%SELFTEST%"
if errorlevel 1 (
    echo.
    echo  [ERROR] The self-test failed. See the messages above.
    goto :fail
)

rem ===========================================================================
rem  Step 4: delete the CSV files of the previous run
rem ===========================================================================
call :progress 4 "Deleting the data files of the previous run"
rem  The names of the output files are obtained from the Python script
set "OUTPUTS="
for /f "delims=" %%f in ('%PY% -B "%SCRIPT%" --list-outputs') do call :append_output "%%f"
if not defined OUTPUTS (
    echo  [ERROR] The list of output files could not be obtained from %SCRIPT%.
    goto :fail
)
if not exist "%DATADIR%" mkdir "%DATADIR%"
set /a "N_DELETED=0"
for %%f in (%OUTPUTS%) do (
    if exist "%DATADIR%\%%f" (
        del /f /q "%DATADIR%\%%f"
        set /a "N_DELETED+=1"
    )
)
echo          Deleted %N_DELETED% old file^(s^) in the folder "%DATADIR%".

rem ===========================================================================
rem  Step 5: run the numerical experiments
rem ===========================================================================
call :progress 5 "Running %SCRIPT%"
echo.
rem  -B: do not write .pyc files (no __pycache__ folders are created)
%PY% -B "%SCRIPT%" --data-dir "%DATADIR%"
if errorlevel 1 (
    echo.
    echo  [ERROR] %SCRIPT% terminated with an error. See the messages above.
    goto :fail
)

rem ===========================================================================
rem  Step 6: check that all expected data files have been written
rem ===========================================================================
call :progress 6 "Checking the new data files"
set /a "N_OK=0, N_BAD=0"
for %%f in (%OUTPUTS%) do call :check_output "%%f"
if %N_BAD% gtr 0 (
    echo  [ERROR] %N_BAD% expected file^(s^) are missing.
    goto :fail
)
echo          All %N_OK% data files have been written to the folder "%DATADIR%".

rem ===========================================================================
rem  Step 7: remove Python cache folders
rem ===========================================================================
call :progress 7 "Removing temporary Python files"
if exist "__pycache__" rd /s /q "__pycache__"
echo          Done.

echo.
echo  ===========================================================================
echo   SUCCESS: the data for the figures and tables have been updated.
echo   Started: %T_START%    Finished: %TIME%
echo   The CSV files are in the folder "%DATADIR%" (see README.md).
echo  ===========================================================================
echo.
if not defined NOPAUSE pause
endlocal
exit /b 0

rem ===========================================================================
rem  Failure exit
rem ===========================================================================
:fail
if exist "__pycache__" rd /s /q "__pycache__"
echo.
echo  ===========================================================================
echo   FAILED: the data files have NOT been updated.
echo  ===========================================================================
echo.
if not defined NOPAUSE pause
endlocal
exit /b 1

rem ===========================================================================
rem  Subroutines
rem ===========================================================================

rem ---------------------------------------------------------------------------
rem  :progress <step> "<label>"
rem  Draws a text progress bar, e.g.  [##########----------]  50%  Step 3/6: ...
rem ---------------------------------------------------------------------------
:progress
setlocal EnableDelayedExpansion
set /a "PCT=%~1*100/STEPS, FILLED=%~1*40/STEPS, EMPTY=40-FILLED"
set "BAR="
for /l %%i in (1,1,!FILLED!) do set "BAR=!BAR!#"
for /l %%i in (1,1,!EMPTY!) do set "BAR=!BAR!-"
set "PCTTXT=  !PCT!"
echo.
echo  [!BAR!] !PCTTXT:~-3!%%  Step %~1/!STEPS!: %~2
endlocal
exit /b 0

rem ---------------------------------------------------------------------------
rem  :append_output "<file name>"   -  append a file name to OUTPUTS
rem ---------------------------------------------------------------------------
:append_output
set "OUTPUTS=%OUTPUTS% %~1"
exit /b 0

rem ---------------------------------------------------------------------------
rem  :check_output "<file name>"
rem  Counts the expected output files that exist (N_OK) and are missing (N_BAD).
rem ---------------------------------------------------------------------------
:check_output
if exist "%DATADIR%\%~1" (
    set /a "N_OK+=1"
    echo          %DATADIR%\%~1 ... OK
) else (
    set /a "N_BAD+=1"
    echo          %DATADIR%\%~1 ... MISSING
)
exit /b 0
