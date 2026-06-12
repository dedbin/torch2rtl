@echo off
set "YOSYSHQ_ROOT=%~dp0..\oss-cad-suite\"
call "%YOSYSHQ_ROOT%environment.bat" >nul
set "VERILATOR_ROOT=%YOSYSHQ_ROOT%share\verilator"
"%YOSYSHQ_ROOT%bin\verilator_bin.exe" %*
