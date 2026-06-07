Set-Location -LiteralPath "E:\BAISYS_QUANT\BAISYS_QUANT"
$env:PYTHONIOENCODING = "utf-8"
$logFile = "E:\BAISYS_QUANT\BAISYS_QUANT\sensitivity_results\first_pass.log"
python sensitivity_runner.py --scan first_pass --resume *>> $logFile
