"""
开发入口 — 直接运行: python app.py [--data ./data] [--port 8765]
实际代码在 src/web_labeling/app.py，这里只做转发，避免维护两份代码。
"""
import sys
import os

# 让 import web_labeling 能找到 src/ 下的包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from web_labeling.app import main  # noqa: E402

if __name__ == "__main__":
    main()
