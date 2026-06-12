@echo off
set "YOSYSHQ_ROOT=%~dp0..\oss-cad-suite\"
call "%YOSYSHQ_ROOT%environment.bat" >nul
"%YOSYSHQ_ROOT%bin\iverilog.exe" -B "%YOSYSHQ_ROOT%lib\ivl" %*
