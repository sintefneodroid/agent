@echo off
cls
del /q "logs\*"
del /q "models\*"
mkdir models
mkdir logs
type nul >> models/.trackfolder
type nul >> logs/.trackfolder
pause