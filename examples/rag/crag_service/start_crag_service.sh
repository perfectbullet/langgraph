#!/bin/bash
# CRAG 服务启动脚本 (Linux/Mac)

echo "🚀 启动 CRAG 服务..."

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo "✓ 激活虚拟环境..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✓ 激活虚拟环境..."
    source venv/bin/activate
fi

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，使用默认配置"
    echo "建议从 .env.example 复制并修改配置:"
    echo "   cp .env.example .env"
fi

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动服务
echo "✓ 启动 FastAPI 服务..."
python crag_service.py
