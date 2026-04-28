# Web Labeling - 图片标注系统

基于浏览器的图片标注工具，支持多种标注格式（X-AnyLabeling JSON、VOC XML、COCO JSON、YOLO TXT）。

## 安装

从 GitHub 安装（推荐）：

```bash
pip install git+https://github.com/zhouchanggeng/web-labeling.git
```

或克隆后本地安装：

```bash
git clone https://github.com/zhouchanggeng/web-labeling.git
cd web-labeling
pip install .
```

## 使用

```bash
# 最简启动
web-labeling --data /path/to/images

# 指定标注目录和格式
web-labeling --data /path/to/images --labels /path/to/labels --format voc

# 指定端口
web-labeling --data /path/to/images --port 9000
```

也可以直接运行源码：

```bash
python app.py --data /path/to/images
```

启动后浏览器打开 `http://localhost:8765`。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `./data` | 图片目录 |
| `--labels` | 与图片目录相同 | 标注文件目录 |
| `--format` | `auto` | 标注格式：auto / json / voc / coco / yolo |
| `--port` | `8765` | 端口号 |
| `--host` | `0.0.0.0` | 监听地址 |

## 功能

- 支持多种标注格式读取：X-AnyLabeling JSON、VOC XML、COCO JSON、YOLO TXT
- 前端可切换图片目录、标注目录和标注格式（点击工具栏 📂 按钮）
- 矩形框标注（快捷键 R）
- 多边形标注（快捷键 P，双击/右键完成）
- 拖拽移动和调整标注框
- 标签筛选：点击侧边栏标签可筛选包含该标签的图片
- 标签输入历史和自动补全
- 滚轮缩放，Alt+拖拽平移

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| a / d | 上一张 / 下一张 |
| r | 矩形工具 |
| p | 多边形工具 |
| v | 选择模式 |
| Delete | 删除选中标注 |
| Ctrl+s | 保存 |
| f | 适应窗口 |
| +/- | 缩放 |
| Esc | 取消绘制 / 取消选择 |

## 支持的标注格式

| 格式 | 文件类型 | 说明 |
|------|----------|------|
| X-AnyLabeling | `*.json`（与图片同名） | 默认格式，保存时也使用此格式 |
| VOC | `*.xml`（与图片同名） | Pascal VOC 格式 |
| COCO | 单个 `*.json`（含 images + annotations） | COCO 目标检测格式 |
| YOLO | `*.txt` + `classes.txt` | 归一化坐标，自动转换为像素坐标 |

保存标注时统一输出为 X-AnyLabeling JSON 格式。

## 发布到 PyPI（可选）

如果需要通过 `pip install web-labeling` 直接安装：

```bash
pip install build twine
python -m build
twine upload dist/*
```
