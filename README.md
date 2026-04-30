# Web Labeling - 图片标注系统

基于浏览器的图片标注工具，支持多种标注格式（X-AnyLabeling JSON、VOC XML、COCO JSON、YOLO TXT），集成 SAM3 AI 辅助标注和模型评估功能。

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
- 撤销操作（Ctrl+Z），支持最多 50 步
- 自动保存：切换图片时自动保存当前标注，无需手动操作

### 显示控制
- 标注框显示/隐藏切换（单个或全部）
- 标签文字显示/隐藏切换（Aa 按钮）
- 实时显示鼠标在图像上的坐标位置
- 标注列表显示每个框的宽×高像素尺寸

### 性能优化（大量图片场景）
- 懒加载分页：图片列表按需分批加载（每页 200 张），不再一次性拉取全部文件名
- 虚拟滚动：侧边栏图片列表只渲染可见区域的 DOM 节点，万级图片无卡顿
- 服务端过滤：搜索和标签筛选在后端完成，前端只接收匹配结果
- 图片预加载：自动预加载相邻图片，切图更流畅
- 防抖搜索：搜索输入 250ms 防抖，减少无效请求
- 目录扫描缓存：后端缓存图片列表和标签扫描结果，避免重复磁盘 IO

### SAM3 AI 辅助标注
- 集成 Meta SAM3（Segment Anything Model 3）模型
- **文本标注**：输入文本描述（如 `person,car`），AI 自动检测并标注所有匹配目标
- **点标注**：在画布上点击目标位置，AI 生成分割结果（Shift+点击为负样本排除区域）
- **批量推理**：一键对所有图片进行 AI 自动标注，支持进度显示
- 支持输出矩形框或多边形 mask
- 模型首次使用时自动加载，需将 `sam3.pt` 和 `bpe_simple_vocab_16e6.txt.gz` 放入 `models/` 目录

### 模型评估
- 输入 GT 标注目录，支持同时评估多个模型的推理结果
- 各类别 PR 曲线（Precision-Recall）
- 各类别 F1 曲线（F1 vs Confidence）
- 混淆矩阵热力图（含背景类）
- 各类别 AP@50 指标及 mAP@50 汇总
- 评估过程实时进度条显示

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

## 项目结构

```
web-labeling/
├── app.py                          # 开发入口（转发到 src/web_labeling/app.py）
├── pyproject.toml                  # pip 包配置
├── requirements.txt                # 依赖
├── README.md
└── src/web_labeling/               # 唯一的业务代码（pip 包和开发共用）
    ├── __init__.py
    ├── app.py                      # Flask 后端（API + 标注读写 + SAM3 + 评估）
    └── static/
        ├── index.html              # 前端页面
        ├── labeler.js              # 前端逻辑（标注、虚拟滚动、懒加载等）
        └── style.css               # 样式
```

> 只维护一份代码：`python app.py` 开发运行和 `pip install` 后的 `web-labeling` 命令使用同一套 `src/web_labeling/` 下的文件。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| A / D | 上一张 / 下一张 |
| J | 随机跳转一张图片 |
| R | 矩形工具 |
| P | 多边形工具 |
| V | 选择模式 |
| F | 适应窗口 |
| Delete / Backspace | 删除选中标注 |
| Ctrl+Z | 撤销 |
| Ctrl+S | 保存 |
| +/- | 缩放 |
| Esc | 取消绘制 / 取消选择 |
| Alt+拖拽 | 平移画布 |
| 右键 | 上下文菜单 |
| 滚轮 | 缩放 |

## SAM3 模型配置

将以下文件放入项目根目录的 `models/` 文件夹：

| 文件 | 说明 |
|------|------|
| `sam3.pt` | SAM3 模型权重（约 3.4GB） |
| `bpe_simple_vocab_16e6.txt.gz` | BPE 文本分词器词表 |

同时需要安装 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) 的依赖（PyTorch 2.7+、CUDA 12.6+）。SAM3 功能为可选，未配置模型时其他标注功能正常使用。

## 支持的标注格式

| 格式 | 文件类型 | 说明 |
|------|----------|------|
| X-AnyLabeling | `*.json`（与图片同名） | 默认格式，保存时也使用此格式 |
| VOC | `*.xml`（与图片同名） | Pascal VOC 格式 |
| COCO | 单个 `*.json`（含 images + annotations） | COCO 目标检测格式 |
| YOLO | `*.txt` + `classes.txt` | 归一化坐标，自动转换为像素坐标 |
