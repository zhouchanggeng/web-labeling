import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request, abort
from PIL import Image

app = Flask(__name__, static_folder="static")

DATA_DIR = os.environ.get("LABEL_DATA_DIR", "./data")
LABEL_DIR = None
LABEL_FORMAT = "auto"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ============ Caches ============
_format_cache = None        # detected format string
_img_size_cache = {}        # {img_name: (w, h)}
_label_cache = None         # {label: [img_name, ...]}
_coco_cache = {}            # {path: parsed}
_coco_fname_index = {}      # {file_name: (cache_key, img_id)}
_yolo_classes = {}


def _invalidate_caches():
    global _format_cache, _img_size_cache, _label_cache
    global _coco_cache, _coco_fname_index, _yolo_classes
    _format_cache = None
    _img_size_cache = {}
    _label_cache = None
    _coco_cache = {}
    _coco_fname_index = {}
    _yolo_classes = {}


def _img_dir():
    return Path(DATA_DIR)


def _label_dir():
    return Path(LABEL_DIR) if LABEL_DIR else Path(DATA_DIR)


def _list_images():
    d = _img_dir()
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)


def _get_img_size(img_name):
    if img_name in _img_size_cache:
        return _img_size_cache[img_name]
    img_path = _img_dir() / img_name
    if img_path.exists():
        with Image.open(img_path) as im:
            size = im.size
        _img_size_cache[img_name] = size
        return size
    return (0, 0)


def _empty_annotation(img_name, w=0, h=0):
    return {
        "version": "2.4.0", "flags": {}, "shapes": [],
        "imagePath": img_name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }


# ============ Format detection (cached) ============

def _detect_format():
    global _format_cache
    if _format_cache is not None:
        return _format_cache
    ld = _label_dir()
    if not ld.exists():
        _format_cache = "json"
        return _format_cache
    img_stems = None
    for f in ld.iterdir():
        if f.suffix == ".json":
            if img_stems is None:
                img_stems = {Path(x).stem for x in _list_images()}
            if f.stem not in img_stems:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        head = fh.read(200)
                    if '"images"' in head and '"annotations"' in head:
                        _format_cache = "coco"
                        return _format_cache
                except Exception:
                    pass
    if any(f.suffix == ".xml" for f in ld.iterdir()):
        _format_cache = "voc"
        return _format_cache
    cls_file = ld / "classes.txt"
    txt_files = [f for f in ld.iterdir() if f.suffix == ".txt" and f.stem != "classes"]
    if txt_files and cls_file.exists():
        _format_cache = "yolo"
        return _format_cache
    if txt_files:
        try:
            with open(txt_files[0], "r") as fh:
                parts = fh.readline().strip().split()
                if len(parts) == 5 and all(_is_number(p) for p in parts):
                    _format_cache = "yolo"
                    return _format_cache
        except Exception:
            pass
    _format_cache = "json"
    return _format_cache


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_current_format():
    if LABEL_FORMAT == "auto":
        return _detect_format()
    return LABEL_FORMAT


# ============ VOC reader ============

def _read_voc(img_name):
    xml_path = _label_dir() / (Path(img_name).stem + ".xml")
    if not xml_path.exists():
        w, h = _get_img_size(img_name)
        return _empty_annotation(img_name, w, h)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width", "0")) if size is not None else 0
    h = int(size.findtext("height", "0")) if size is not None else 0
    shapes = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "")
        difficult = obj.findtext("difficult", "0") == "1"
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            shapes.append({
                "label": label, "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None, "description": "", "difficult": difficult,
                "shape_type": "rectangle", "flags": {}, "attributes": {},
                "kie_linking": [],
            })
    return {
        "version": "2.4.0", "flags": {}, "shapes": shapes,
        "imagePath": img_name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }


# ============ COCO reader (indexed) ============

def _load_coco():
    global _coco_cache, _coco_fname_index
    if _coco_cache:
        return
    ld = _label_dir()
    for f in ld.iterdir():
        if f.suffix != ".json":
            continue
        fp = str(f.resolve())
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "images" not in data or "annotations" not in data:
                continue
            cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
            img_map = {im["id"]: im for im in data["images"]}
            ann_by_img = {}
            for ann in data["annotations"]:
                ann_by_img.setdefault(ann["image_id"], []).append(ann)
            _coco_cache[fp] = {"cat_map": cat_map, "img_map": img_map, "ann_by_img": ann_by_img}
            # Build filename index for O(1) lookup
            for img_id, im_info in img_map.items():
                fname = im_info.get("file_name", "")
                if fname:
                    _coco_fname_index[fname] = (fp, img_id)
        except Exception:
            pass


def _read_coco(img_name):
    _load_coco()
    entry = _coco_fname_index.get(img_name)
    if not entry:
        w, h = _get_img_size(img_name)
        return _empty_annotation(img_name, w, h)
    cache_key, img_id = entry
    cache = _coco_cache[cache_key]
    im_info = cache["img_map"][img_id]
    w, h = im_info.get("width", 0), im_info.get("height", 0)
    shapes = []
    for ann in cache["ann_by_img"].get(img_id, []):
        cat_name = cache["cat_map"].get(ann.get("category_id"), "unknown")
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, bw, bh = bbox
            shapes.append({
                "label": cat_name, "points": [[x, y], [x + bw, y + bh]],
                "group_id": None, "description": "",
                "difficult": ann.get("iscrowd", 0) == 1,
                "shape_type": "rectangle", "flags": {}, "attributes": {},
                "kie_linking": [],
            })
        seg = ann.get("segmentation")
        if seg and isinstance(seg, list) and len(seg) > 0 and not bbox:
            for poly in seg:
                pts = [[poly[i], poly[i+1]] for i in range(0, len(poly), 2)]
                shapes.append({
                    "label": cat_name, "points": pts,
                    "group_id": None, "description": "", "difficult": False,
                    "shape_type": "polygon", "flags": {}, "attributes": {},
                    "kie_linking": [],
                })
    return {
        "version": "2.4.0", "flags": {}, "shapes": shapes,
        "imagePath": img_name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }


# ============ YOLO reader ============

def _load_yolo_classes():
    global _yolo_classes
    if _yolo_classes:
        return
    cls_file = _label_dir() / "classes.txt"
    if cls_file.exists():
        with open(cls_file, "r", encoding="utf-8") as f:
            _yolo_classes = {i: line.strip() for i, line in enumerate(f) if line.strip()}


def _read_yolo(img_name):
    _load_yolo_classes()
    txt_path = _label_dir() / (Path(img_name).stem + ".txt")
    w, h = _get_img_size(img_name)
    if not txt_path.exists() or w == 0 or h == 0:
        return _empty_annotation(img_name, w, h)
    shapes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            label = _yolo_classes.get(cls_id, str(cls_id))
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            shapes.append({
                "label": label, "points": [[x1, y1], [x2, y2]],
                "group_id": None, "description": "", "difficult": False,
                "shape_type": "rectangle", "flags": {}, "attributes": {},
                "kie_linking": [],
            })
    return {
        "version": "2.4.0", "flags": {}, "shapes": shapes,
        "imagePath": img_name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }


# ============ JSON reader ============

def _read_json(img_name):
    jp = _label_dir() / (Path(img_name).stem + ".json")
    if jp.exists():
        with open(jp, "r", encoding="utf-8") as f:
            return json.load(f)
    w, h = _get_img_size(img_name)
    return _empty_annotation(img_name, w, h)


# ============ Unified reader ============

def _read_annotation(img_name):
    fmt = _get_current_format()
    if fmt == "voc":
        return _read_voc(img_name)
    elif fmt == "coco":
        return _read_coco(img_name)
    elif fmt == "yolo":
        return _read_yolo(img_name)
    else:
        return _read_json(img_name)


# ============ Labels scan (cached) ============

def _scan_labels():
    global _label_cache
    if _label_cache is not None:
        return _label_cache
    fmt = _get_current_format()
    label_map = {}
    images_set = set(_list_images())

    if fmt == "json":
        ld = _label_dir()
        for jp in ld.glob("*.json"):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                img_name = ann.get("imagePath", jp.stem)
                for s in ann.get("shapes", []):
                    lbl = s.get("label", "")
                    if lbl:
                        label_map.setdefault(lbl, []).append(img_name)
            except Exception:
                continue
    elif fmt == "voc":
        ld = _label_dir()
        for xp in ld.glob("*.xml"):
            try:
                tree = ET.parse(xp)
                root = tree.getroot()
                fname = root.findtext("filename", xp.stem)
                for obj in root.findall("object"):
                    lbl = obj.findtext("name", "")
                    if lbl:
                        label_map.setdefault(lbl, []).append(fname)
            except Exception:
                continue
    elif fmt == "coco":
        _load_coco()
        for cache in _coco_cache.values():
            for img_id, anns in cache["ann_by_img"].items():
                im_info = cache["img_map"].get(img_id, {})
                fname = im_info.get("file_name", "")
                for ann in anns:
                    lbl = cache["cat_map"].get(ann.get("category_id"), "")
                    if lbl:
                        label_map.setdefault(lbl, []).append(fname)
    elif fmt == "yolo":
        _load_yolo_classes()
        ld = _label_dir()
        for tp in ld.glob("*.txt"):
            if tp.stem == "classes":
                continue
            img_name = None
            for ext in IMG_EXTS:
                candidate = tp.stem + ext
                if candidate in images_set:
                    img_name = candidate
                    break
            if not img_name:
                continue
            try:
                with open(tp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            lbl = _yolo_classes.get(cls_id, str(cls_id))
                            label_map.setdefault(lbl, []).append(img_name)
            except Exception:
                continue

    for lbl in label_map:
        label_map[lbl] = sorted(set(label_map[lbl]))
    _label_cache = label_map
    return _label_cache


# ============ API routes ============

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/images")
def api_images():
    return jsonify(_list_images())


@app.route("/api/image/<path:name>")
def api_image(name):
    d = _img_dir()
    fp = d / name
    if not fp.exists():
        abort(404)
    return send_from_directory(str(d), name)


@app.route("/api/annotation/<path:name>", methods=["GET"])
def api_get_annotation(name):
    return jsonify(_read_annotation(name))


@app.route("/api/annotation/<path:name>", methods=["POST"])
def api_save_annotation(name):
    global _label_cache
    data = request.get_json(force=True)
    jp = _label_dir() / (Path(name).stem + ".json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _label_cache = None  # Invalidate label cache on save
    return jsonify({"ok": True})


@app.route("/api/labels")
def api_labels():
    return jsonify(_scan_labels())


@app.route("/api/datadir", methods=["GET"])
def api_get_datadir():
    return jsonify({
        "path": str(Path(DATA_DIR).resolve()),
        "label_path": str(Path(LABEL_DIR).resolve()) if LABEL_DIR else "",
        "format": LABEL_FORMAT,
    })


@app.route("/api/datadir", methods=["POST"])
def api_set_datadir():
    global DATA_DIR, LABEL_DIR, LABEL_FORMAT
    data = request.get_json(force=True)
    p = data.get("path", "").strip()
    if not p:
        return jsonify({"ok": False, "error": "路径不能为空"}), 400
    expanded = os.path.expanduser(p)
    if not os.path.isdir(expanded):
        return jsonify({"ok": False, "error": f"目录不存在: {expanded}"}), 400
    DATA_DIR = expanded
    lp = data.get("label_path", "").strip()
    if lp:
        lp_expanded = os.path.expanduser(lp)
        if os.path.isdir(lp_expanded):
            LABEL_DIR = lp_expanded
        else:
            return jsonify({"ok": False, "error": f"标注目录不存在: {lp_expanded}"}), 400
    else:
        LABEL_DIR = None
    fmt = data.get("format", "auto").strip()
    LABEL_FORMAT = fmt if fmt in ("auto", "json", "voc", "coco", "yolo") else "auto"
    _invalidate_caches()
    return jsonify({"ok": True, "path": str(Path(DATA_DIR).resolve())})


@app.route("/api/browse")
def api_browse():
    p = request.args.get("path", "~")
    expanded = Path(os.path.expanduser(p)).resolve()
    if not expanded.is_dir():
        return jsonify({"path": str(expanded), "dirs": [], "error": "不是有效目录"})
    dirs = []
    try:
        for f in sorted(expanded.iterdir()):
            if f.is_dir() and not f.name.startswith('.'):
                # Only count images for immediate children, skip on error
                try:
                    img_count = sum(1 for x in f.iterdir() if x.suffix.lower() in IMG_EXTS)
                except (PermissionError, OSError):
                    img_count = 0
                dirs.append({"name": f.name, "img_count": img_count})
    except PermissionError:
        return jsonify({"path": str(expanded), "dirs": [], "error": "无权限访问"})
    return jsonify({"path": str(expanded), "dirs": dirs})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Web Labeling Server")
    parser.add_argument("--data", default="./data", help="图片数据目录")
    parser.add_argument("--labels", default=None, help="标注文件目录（默认与图片目录相同）")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "json", "voc", "coco", "yolo"], help="标注格式")
    parser.add_argument("--port", type=int, default=8765, help="端口号")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    args = parser.parse_args()
    DATA_DIR = args.data
    LABEL_DIR = args.labels
    LABEL_FORMAT = args.format
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"数据目录: {os.path.abspath(DATA_DIR)}")
    if LABEL_DIR:
        print(f"标注目录: {os.path.abspath(LABEL_DIR)}")
    print(f"标注格式: {LABEL_FORMAT}")
    print(f"服务地址: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
