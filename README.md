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

### 基础标注
- 矩形框标注（快捷键 R）
- 多边形标注（快捷键 P，双击/右键完成）
- 拖拽移动和调整标注框
- 鼠标悬停自动选中标注框
- 右键上下文菜单：更改标签、删除、复制、置顶/置底
- 标签输入时显示已有标签的快捷选择按钮
- 标签筛选：点击侧边栏标签可筛选包含该标签的图片（实时统计）

### 显示控制
- 标注框显示/隐藏切换（单个或全部）
- 标签文字显示/隐藏切换（Aa 按钮）
- 实时显示鼠标在图像上的坐标位置
- 标注列表显示每个框的宽×高像素尺寸

### 模型对比
- 支持加载两个模型的推理结果目录进行对比
- 基于 IoU 匹配分析差异：仅A检出、仅B检出、类别不一致、完全一致
- 按差异类型过滤图片，图片列表颜色标记差异类型
- 画布上叠加显示两套结果（红色=模型A，蓝色=模型B）
- 显示具体模型名称（从目录名提取）
- 对比结果缓存到文件，关闭后再次打开无需重新计算

### 冲突检测
- 一键检测同一张图中类别不同但高度重合的标注框（IoU ≥ 0.5）
- 自动过滤显示有冲突的图片，方便逐张修正

### 标注统计分析
- 类别分布：数量、占比、柱状图
- 框尺寸分布：小/中/大目标统计
- 各类别框尺寸统计：归一化平均宽高、范围
- 宽高散点图可视化

### 多格式支持
- 支持读取：X-AnyLabeling JSON、VOC XML、COCO JSON、YOLO TXT
- 自动检测标注格式
- 保存统一输出为 X-AnyLabeling JSON 格式
- 前端可切换图片目录、标注目录和标注格式（📂 按钮）
- 文件夹浏览器支持点击选择目录

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| A / D | 上一张 / 下一张 |
| R | 矩形工具 |
| P | 多边形工具 |
| V | 选择模式 |
| F | 适应窗口 |
| Delete / Backspace | 删除选中标注 |
| Ctrl+S | 保存 |
| +/- | 缩放 |
| Esc | 取消绘制 / 取消选择 |
| Alt+拖拽 | 平移画布 |
| 右键 | 上下文菜单 |
| 滚轮 | 缩放 |

## 支持的标注格式

| 格式 | 文件类型 | 说明 |
|------|----------|------|
| X-AnyLabeling | `*.json`（与图片同名） | 默认格式，保存时也使用此格式 |
| VOC | `*.xml`（与图片同名） | Pascal VOC 格式 |
| COCO | 单个 `*.json`（含 images + annotations） | COCO 目标检测格式 |
| YOLO | `*.txt` + `classes.txt` | 归一化坐标，自动转换为像素坐标 |
