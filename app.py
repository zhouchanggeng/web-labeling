import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, send_from_directory, send_file, jsonify, request, abort
from PIL import Image

app = Flask(__name__, static_folder="static")

DATA_DIR = os.environ.get("LABEL_DATA_DIR", "./data")
LABEL_DIR = None
LABEL_FORMAT = "auto"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ============ Compare mode state ============
COMPARE_A_DIR = None
COMPARE_A_FORMAT = "auto"
COMPARE_B_DIR = None
COMPARE_B_FORMAT = "auto"
_compare_result = None  # cached comparison result

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
    global _compare_result
    _format_cache = None
    _img_size_cache = {}
    _label_cache = None
    _coco_cache = {}
    _coco_fname_index = {}
    _yolo_classes = {}
    _compare_result = None


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
        with Image.open(img_path.resolve()) as im:
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


# ============ Conflict detection ============

def _find_conflicts_in_annotation(ann, iou_thresh=0.5):
    """Find shapes with different labels but high IoU overlap in one annotation."""
    shapes = ann.get("shapes", [])
    conflicts = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            sa, sb = shapes[i], shapes[j]
            la, lb = sa.get("label", ""), sb.get("label", "")
            if la == lb or not la or not lb:
                continue
            box_a = _shape_box(sa)
            box_b = _shape_box(sb)
            if not box_a or not box_b:
                continue
            v = _iou(box_a, box_b)
            if v >= iou_thresh:
                conflicts.append({
                    "a": {"label": la, "bbox": box_a, "idx": i},
                    "b": {"label": lb, "bbox": box_b, "idx": j},
                    "iou": round(v, 4),
                })
    return conflicts


def _scan_all_conflicts(iou_thresh=0.5):
    """Scan all images for label conflicts."""
    images = _list_images()
    result = []
    for img_name in images:
        ann = _read_annotation(img_name)
        conflicts = _find_conflicts_in_annotation(ann, iou_thresh)
        if conflicts:
            result.append({"image": img_name, "conflicts": conflicts})
    return result


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
    real = fp.resolve()
    return send_file(real)


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


@app.route("/api/conflicts")
def api_conflicts():
    thresh = request.args.get("iou", 0.5, type=float)
    result = _scan_all_conflicts(thresh)
    return jsonify({"ok": True, "conflicts": result,
                    "total_images": len(result),
                    "total_pairs": sum(len(r["conflicts"]) for r in result)})


@app.route("/api/analysis")
def api_analysis():
    """Analyze all annotations: class distribution, box sizes, aspect ratios."""
    images = _list_images()
    class_counts = {}
    box_widths = []
    box_heights = []
    box_areas = []
    aspect_ratios = []
    boxes_per_image = []
    total_boxes = 0
    images_with_ann = 0
    class_size_data = {}  # {label: [(w, h), ...]}

    for img_name in images:
        ann = _read_annotation(img_name)
        shapes = ann.get("shapes", [])
        img_w = ann.get("imageWidth", 0) or 1
        img_h = ann.get("imageHeight", 0) or 1
        boxes_per_image.append(len(shapes))
        if shapes:
            images_with_ann += 1
        for s in shapes:
            label = s.get("label", "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1
            total_boxes += 1
            box = _shape_box(s)
            if box:
                w = abs(box[1][0] - box[0][0])
                h = abs(box[1][1] - box[0][1])
                # Normalize to image size
                nw = w / img_w
                nh = h / img_h
                box_widths.append(nw)
                box_heights.append(nh)
                box_areas.append(nw * nh)
                if h > 0:
                    aspect_ratios.append(w / h)
                class_size_data.setdefault(label, []).append([nw, nh])

    # Size distribution buckets (small/medium/large by area)
    small = sum(1 for a in box_areas if a < 0.01)
    medium = sum(1 for a in box_areas if 0.01 <= a < 0.1)
    large = sum(1 for a in box_areas if a >= 0.1)

    # Per-class stats
    class_stats = {}
    for label, sizes in class_size_data.items():
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        class_stats[label] = {
            "count": len(sizes),
            "avg_w": round(sum(ws) / len(ws), 4) if ws else 0,
            "avg_h": round(sum(hs) / len(hs), 4) if hs else 0,
            "min_w": round(min(ws), 4) if ws else 0,
            "max_w": round(max(ws), 4) if ws else 0,
            "min_h": round(min(hs), 4) if hs else 0,
            "max_h": round(max(hs), 4) if hs else 0,
        }

    return jsonify({
        "ok": True,
        "total_images": len(images),
        "images_with_ann": images_with_ann,
        "total_boxes": total_boxes,
        "class_counts": class_counts,
        "class_stats": class_stats,
        "size_distribution": {"small": small, "medium": medium, "large": large},
        "avg_boxes_per_image": round(sum(boxes_per_image) / len(boxes_per_image), 2) if boxes_per_image else 0,
        "box_widths": [round(w, 4) for w in box_widths[:2000]],
        "box_heights": [round(h, 4) for h in box_heights[:2000]],
    })


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


# ============ Compare mode helpers ============

def _read_annotation_from(label_dir, fmt, img_name):
    """Read annotation for img_name from a specific label directory with given format."""
    # Temporarily swap globals to reuse existing readers
    global LABEL_DIR, LABEL_FORMAT
    global _format_cache, _coco_cache, _coco_fname_index, _yolo_classes
    old_ld, old_fmt = LABEL_DIR, LABEL_FORMAT
    old_fc, old_cc, old_ci, old_yc = _format_cache, _coco_cache, _coco_fname_index, _yolo_classes
    try:
        LABEL_DIR = label_dir
        LABEL_FORMAT = fmt
        _format_cache = None
        _coco_cache = {}
        _coco_fname_index = {}
        _yolo_classes = {}
        return _read_annotation(img_name)
    finally:
        LABEL_DIR = old_ld
        LABEL_FORMAT = old_fmt
        _format_cache = old_fc
        _coco_cache = old_cc
        _coco_fname_index = old_ci
        _yolo_classes = old_yc


def _iou(box_a, box_b):
    """Compute IoU between two boxes [[x1,y1],[x2,y2]]."""
    ax1, ay1 = box_a[0]
    ax2, ay2 = box_a[1]
    bx1, by1 = box_b[0]
    bx2, by2 = box_b[1]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _shape_box(s):
    """Get [[x1,y1],[x2,y2]] from a shape."""
    pts = s.get("points", [])
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


def _compare_annotations(ann_a, ann_b, iou_thresh=0.3):
    """
    Compare two annotations for the same image.
    Returns diff info:
      - a_only: shapes only in A (B missed)
      - b_only: shapes only in B (A missed)
      - mismatch: matched shapes with different labels
      - matched: shapes matched with same label
    """
    shapes_a = ann_a.get("shapes", [])
    shapes_b = ann_b.get("shapes", [])
    used_b = set()
    matched = []
    mismatch = []
    a_only = []

    for sa in shapes_a:
        box_a = _shape_box(sa)
        if not box_a:
            continue
        best_iou = 0
        best_j = -1
        for j, sb in enumerate(shapes_b):
            if j in used_b:
                continue
            box_b = _shape_box(sb)
            if not box_b:
                continue
            v = _iou(box_a, box_b)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            used_b.add(best_j)
            sb = shapes_b[best_j]
            if sa.get("label") == sb.get("label"):
                matched.append({"a": sa, "b": sb, "iou": best_iou})
            else:
                mismatch.append({"a": sa, "b": sb, "iou": best_iou})
        else:
            a_only.append(sa)

    b_only = [sb for j, sb in enumerate(shapes_b) if j not in used_b]
    return {
        "a_only": a_only,
        "b_only": b_only,
        "mismatch": mismatch,
        "matched": matched,
    }


def _compare_cache_path():
    """Generate a cache file path based on data dir + A/B dirs."""
    key = f"{COMPARE_A_DIR}|{COMPARE_B_DIR}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    a_name = os.path.basename(COMPARE_A_DIR)
    b_name = os.path.basename(COMPARE_B_DIR)
    fname = f".compare_{a_name}_vs_{b_name}_{h}.json"
    return _img_dir() / fname


def _run_compare():
    """Run full comparison between A and B directories. Uses file cache."""
    global _compare_result
    if _compare_result is not None:
        return _compare_result
    if not COMPARE_A_DIR or not COMPARE_B_DIR:
        return None

    # Try loading from cache file
    cache_path = _compare_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Verify it matches current A/B dirs
            if cached.get("_a_dir") == COMPARE_A_DIR and cached.get("_b_dir") == COMPARE_B_DIR:
                _compare_result = cached
                return _compare_result
        except Exception:
            pass

    images = _list_images()
    result = {
        "images": {},       # {img_name: diff_info}
        "a_only_imgs": [],  # images where A has detections B doesn't
        "b_only_imgs": [],  # images where B has detections A doesn't
        "mismatch_imgs": [],  # images with label mismatches
        "both_imgs": [],    # images where both agree
        "summary": {},
    }
    total_a_only = 0
    total_b_only = 0
    total_mismatch = 0
    total_matched = 0

    for img_name in images:
        ann_a = _read_annotation_from(COMPARE_A_DIR, COMPARE_A_FORMAT, img_name)
        ann_b = _read_annotation_from(COMPARE_B_DIR, COMPARE_B_FORMAT, img_name)
        diff = _compare_annotations(ann_a, ann_b)
        result["images"][img_name] = diff
        has_a_only = len(diff["a_only"]) > 0
        has_b_only = len(diff["b_only"]) > 0
        has_mismatch = len(diff["mismatch"]) > 0
        if has_a_only:
            result["a_only_imgs"].append(img_name)
        if has_b_only:
            result["b_only_imgs"].append(img_name)
        if has_mismatch:
            result["mismatch_imgs"].append(img_name)
        if not has_a_only and not has_b_only and not has_mismatch:
            result["both_imgs"].append(img_name)
        total_a_only += len(diff["a_only"])
        total_b_only += len(diff["b_only"])
        total_mismatch += len(diff["mismatch"])
        total_matched += len(diff["matched"])

    result["summary"] = {
        "total_images": len(images),
        "a_only_count": total_a_only,
        "b_only_count": total_b_only,
        "mismatch_count": total_mismatch,
        "matched_count": total_matched,
    }
    _compare_result = result

    # Save to cache file
    try:
        result["_a_dir"] = COMPARE_A_DIR
        result["_b_dir"] = COMPARE_B_DIR
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception:
        pass

    return result


# ============ Compare API routes ============

@app.route("/api/compare/setup", methods=["POST"])
def api_compare_setup():
    global COMPARE_A_DIR, COMPARE_A_FORMAT, COMPARE_B_DIR, COMPARE_B_FORMAT, _compare_result
    data = request.get_json(force=True)
    a_path = data.get("a_path", "").strip()
    b_path = data.get("b_path", "").strip()
    if not a_path or not b_path:
        return jsonify({"ok": False, "error": "两个标注目录都不能为空"}), 400
    a_exp = os.path.expanduser(a_path)
    b_exp = os.path.expanduser(b_path)
    if not os.path.isdir(a_exp):
        return jsonify({"ok": False, "error": f"模型A目录不存在: {a_exp}"}), 400
    if not os.path.isdir(b_exp):
        return jsonify({"ok": False, "error": f"模型B目录不存在: {b_exp}"}), 400
    COMPARE_A_DIR = a_exp
    COMPARE_A_FORMAT = data.get("a_format", "auto")
    COMPARE_B_DIR = b_exp
    COMPARE_B_FORMAT = data.get("b_format", "auto")
    _compare_result = None
    result = _run_compare()
    a_name = os.path.basename(COMPARE_A_DIR)
    b_name = os.path.basename(COMPARE_B_DIR)
    return jsonify({"ok": True, "summary": result["summary"],
                    "a_only_imgs": result["a_only_imgs"],
                    "b_only_imgs": result["b_only_imgs"],
                    "mismatch_imgs": result["mismatch_imgs"],
                    "both_imgs": result["both_imgs"],
                    "a_name": a_name, "b_name": b_name})


@app.route("/api/compare/result")
def api_compare_result():
    result = _run_compare()
    if not result:
        return jsonify({"ok": False, "error": "未设置对比模式"}), 400
    a_name = os.path.basename(COMPARE_A_DIR) if COMPARE_A_DIR else ""
    b_name = os.path.basename(COMPARE_B_DIR) if COMPARE_B_DIR else ""
    return jsonify({"ok": True, "summary": result["summary"],
                    "a_only_imgs": result["a_only_imgs"],
                    "b_only_imgs": result["b_only_imgs"],
                    "mismatch_imgs": result["mismatch_imgs"],
                    "both_imgs": result["both_imgs"],
                    "a_name": a_name, "b_name": b_name})


@app.route("/api/compare/annotation/<path:name>")
def api_compare_annotation(name):
    if not COMPARE_A_DIR or not COMPARE_B_DIR:
        return jsonify({"ok": False, "error": "未设置对比模式"}), 400
    ann_a = _read_annotation_from(COMPARE_A_DIR, COMPARE_A_FORMAT, name)
    ann_b = _read_annotation_from(COMPARE_B_DIR, COMPARE_B_FORMAT, name)
    diff = _compare_annotations(ann_a, ann_b)
    return jsonify({
        "ok": True,
        "a": ann_a,
        "b": ann_b,
        "diff": {
            "a_only": diff["a_only"],
            "b_only": diff["b_only"],
            "mismatch": diff["mismatch"],
            "matched": diff["matched"],
        }
    })


@app.route("/api/compare/close", methods=["POST"])
def api_compare_close():
    global COMPARE_A_DIR, COMPARE_B_DIR, _compare_result
    COMPARE_A_DIR = None
    COMPARE_B_DIR = None
    _compare_result = None
    return jsonify({"ok": True})


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
