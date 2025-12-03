# CRAG 服务启动脚本 (Windows PowerShell)

Write-Host "🚀 启动 CRAG 服务..." -ForegroundColor Green

# 检查虚拟环境
if (Test-Path ".venv") {
    Write-Host "✓ 激活虚拟环境..." -ForegroundColor Green
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path "venv") {
    Write-Host "✓ 激活虚拟环境..." -ForegroundColor Green
    & .\venv\Scripts\Activate.ps1
}

# 检查环境变量文件
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  未找到 .env 文件，使用默认配置" -ForegroundColor Yellow
    Write-Host "建议从 .env.example 复制并修改配置:" -ForegroundColor Yellow
    Write-Host "   Copy-Item .env.example .env" -ForegroundColor Yellow
}

# 设置 Python 路径
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

# 启动服务
Write-Host "✓ 启动 FastAPI 服务..." -ForegroundColor Green
python crag_service.py
