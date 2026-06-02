#!/bin/bash

# 确保遇到错误时停止执行
set -e

# --- 1. 获取前端提交信息 ---
read -p "Enter commit message (default: 'add'): " msg
# 如果输入为空，则使用默认值
FCOMMIT_MSG=${msg:-"add"}

# --- 3. 执行前端推送 ---
# 假设脚本在前端目录下运行
echo "Pushing to GitHub..."
git rm -r --cached .
git add .
# 即使没有文件更改也允许脚本继续
git commit -m "$FCOMMIT_MSG" || echo "No changes to commit in Frontend"
git push origin master

    echo "Done! All repos are up to date."
else
    echo "Error: Frontend directory not found!"
    exit 1
fi