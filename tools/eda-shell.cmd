@echo off
set "YOSYSHQ_ROOT=%~dp0oss-cad-suite\"
call "%YOSYSHQ_ROOT%environment.bat"
set "VERILATOR_ROOT=%YOSYSHQ_ROOT%share\verilator"
set "PATH=%~dp0eda-bin;%PATH%"
cmd /k
