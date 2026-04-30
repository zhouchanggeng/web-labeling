import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, send_from_directory, send_file, jsonify, request, abort
from PIL import Image

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))

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
    # 不读取 classes.txt，直接用数字 ID 作为标签名，避免映射错误
    _yolo_classes = {}


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


# ============ Writers (save back in original format) ============

def _save_yolo(img_name, data):
    """Save annotation back to YOLO .txt format. Label is used directly as class ID (must be numeric)."""
    img_w = data.get("imageWidth", 0) or 1
    img_h = data.get("imageHeight", 0) or 1
    lines = []
    for s in data.get("shapes", []):
        label = s.get("label", "")
        pts = s.get("points", [])
        if len(pts) < 2 or not label:
            continue
        # Label should be the numeric class ID directly
        try:
            cls_id = int(label)
        except ValueError:
            cls_id = 0  # fallback
        # Compute bounding box from points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        # Convert to YOLO normalized format
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt_path = _label_dir() / (Path(img_name).stem + ".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _save_voc(img_name, data):
    """Save annotation back to VOC XML format."""
    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = ""
    ET.SubElement(root, "filename").text = img_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"

    for s in data.get("shapes", []):
        label = s.get("label", "")
        pts = s.get("points", [])
        if len(pts) < 2 or not label:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "difficult").text = "1" if s.get("difficult") else "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(round(min(xs)))
        ET.SubElement(bndbox, "ymin").text = str(round(min(ys)))
        ET.SubElement(bndbox, "xmax").text = str(round(max(xs)))
        ET.SubElement(bndbox, "ymax").text = str(round(max(ys)))

    xml_path = _label_dir() / (Path(img_name).stem + ".xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)


def _save_json(img_name, data):
    """Save annotation as X-AnyLabeling JSON format."""
    jp = _label_dir() / (Path(img_name).stem + ".json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_annotation(img_name, data):
    """Save annotation in the same format as the current source."""
    fmt = _get_current_format()
    if fmt == "yolo":
        _save_yolo(img_name, data)
    elif fmt == "voc":
        _save_voc(img_name, data)
    else:
        # json and coco both save as per-image JSON
        _save_json(img_name, data)


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
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/images")
def api_images():
    images = _list_images()
    # Support lazy-loading pagination with optional search & label filter
    offset = request.args.get("offset", type=int)
    limit = request.args.get("limit", type=int)
    search = request.args.get("search", "", type=str).strip().lower()
    label = request.args.get("label", "", type=str).strip()

    # Apply label filter
    if label:
        labels = _scan_labels()
        allowed = set(labels.get(label, []))
        images = [img for img in images if img in allowed]

    # Apply search filter
    if search:
        images = [img for img in images if search in img.lower()]

    total = len(images)

    if offset is not None and limit is not None:
        page = images[offset:offset + limit]
        return jsonify({"total": total, "offset": offset, "images": page})
    return jsonify(images)


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
    _save_annotation(name, data)
    _label_cache = None  # Invalidate label cache on save
    return jsonify({"ok": True})


@app.route("/api/rotate/<path:name>", methods=["POST"])
def api_rotate(name):
    """Rotate image file and transform annotation coordinates."""
    global _label_cache
    data = request.get_json(force=True)
    angle = data.get("angle", 90)  # +90 = clockwise, -90 = counter-clockwise
    img_path = _img_dir() / name
    if not img_path.exists():
        return jsonify({"ok": False, "error": "图片不存在"}), 404

    # Rotate image file (PIL uses counter-clockwise, so negate)
    with Image.open(img_path) as im:
        old_w, old_h = im.size
        rotated = im.rotate(-angle, expand=True)
        rotated.save(img_path)
        new_w, new_h = rotated.size

    # Clear image size cache
    _img_size_cache.pop(name, None)

    # Clear SAM3 embedding cache for this image
    global _sam3_current_img, _sam3_state
    if _sam3_current_img == name:
        _sam3_current_img = None
        _sam3_state = None

    # Rotate annotation coordinates
    ann = _read_annotation(name)
    if ann.get("shapes"):
        for s in ann["shapes"]:
            pts = s.get("points", [])
            new_pts = []
            for px, py in pts:
                if angle == 90 or angle == -270:
                    new_pts.append([old_h - py, px])
                elif angle == -90 or angle == 270:
                    new_pts.append([py, old_w - px])
                elif abs(angle) == 180:
                    new_pts.append([old_w - px, old_h - py])
                else:
                    new_pts.append([px, py])
            s["points"] = new_pts
        ann["imageWidth"] = new_w
        ann["imageHeight"] = new_h
        _save_annotation(name, ann)
        _label_cache = None

    return jsonify({"ok": True, "width": new_w, "height": new_h})


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


# ============ Evaluation (mAP / PR / F1 / Confusion Matrix) ============

def _evaluate_model(gt_dir, gt_fmt, pred_dir, pred_fmt, iou_thresh=0.5):
    """Evaluate a single model's predictions against ground truth.
    Returns per-class PR curves, F1 curves, AP50, confusion matrix."""
    images = _list_images()
    # Collect all GT and pred boxes: [(img_idx, label, box, score)]
    all_gt = []   # (img_idx, label, box)
    all_pred = [] # (img_idx, label, box, score)
    all_labels = set()

    for img_idx, img_name in enumerate(images):
        gt_ann = _read_annotation_from(gt_dir, gt_fmt, img_name)
        pred_ann = _read_annotation_from(pred_dir, pred_fmt, img_name)
        for s in gt_ann.get("shapes", []):
            box = _shape_box(s)
            if box:
                lbl = s.get("label", "unknown")
                all_labels.add(lbl)
                all_gt.append((img_idx, lbl, box))
        for s in pred_ann.get("shapes", []):
            box = _shape_box(s)
            if box:
                lbl = s.get("label", "unknown")
                score = s.get("score", 1.0)
                all_labels.add(lbl)
                all_pred.append((img_idx, lbl, box, score))

    labels = sorted(all_labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_cls = len(labels)

    # Build per-image GT index
    gt_by_img = {}
    for img_idx, lbl, box in all_gt:
        gt_by_img.setdefault(img_idx, []).append((lbl, box, False))  # (label, box, matched)

    # Sort predictions by score descending
    all_pred.sort(key=lambda x: -x[3])

    # Per-class TP/FP tracking for PR curve
    class_tp_fp = {l: [] for l in labels}  # [(score, is_tp)]
    class_n_gt = {l: 0 for l in labels}
    for _, lbl, _ in all_gt:
        class_n_gt[lbl] = class_n_gt.get(lbl, 0) + 1

    # Confusion matrix: rows=GT, cols=Pred, +1 for background
    conf_matrix = [[0] * (n_cls + 1) for _ in range(n_cls + 1)]

    # Match predictions to GT
    gt_matched = {}  # (img_idx, gt_idx) -> True
    for img_idx, pred_lbl, pred_box, score in all_pred:
        gts = gt_by_img.get(img_idx, [])
        best_iou = 0
        best_gt_idx = -1
        for gi, (gt_lbl, gt_box, _) in enumerate(gts):
            v = _iou(pred_box, gt_box)
            if v > best_iou:
                best_iou = v
                best_gt_idx = gi

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            gt_key = (img_idx, best_gt_idx)
            gt_lbl = gts[best_gt_idx][0]
            if gt_key not in gt_matched:
                gt_matched[gt_key] = True
                if pred_lbl == gt_lbl:
                    class_tp_fp[pred_lbl].append((score, True))
                    conf_matrix[label_to_idx[gt_lbl]][label_to_idx[pred_lbl]] += 1
                else:
                    class_tp_fp[pred_lbl].append((score, False))
                    if gt_lbl in label_to_idx and pred_lbl in label_to_idx:
                        conf_matrix[label_to_idx[gt_lbl]][label_to_idx[pred_lbl]] += 1
            else:
                class_tp_fp[pred_lbl].append((score, False))
                # FP: pred with no unmatched GT -> background column not needed here
        else:
            class_tp_fp[pred_lbl].append((score, False))
            # FP: background predicted as something
            if pred_lbl in label_to_idx:
                conf_matrix[n_cls][label_to_idx[pred_lbl]] += 1

    # Missed GT -> background row
    for img_idx, gts in gt_by_img.items():
        for gi, (gt_lbl, _, _) in enumerate(gts):
            if (img_idx, gi) not in gt_matched:
                if gt_lbl in label_to_idx:
                    conf_matrix[label_to_idx[gt_lbl]][n_cls] += 1

    # Compute PR curves and AP per class
    pr_curves = {}
    f1_curves = {}
    ap_per_class = {}

    for lbl in labels:
        entries = class_tp_fp[lbl]
        n_gt = class_n_gt.get(lbl, 0)
        if n_gt == 0:
            pr_curves[lbl] = {"precision": [], "recall": [], "confidence": []}
            f1_curves[lbl] = {"f1": [], "confidence": []}
            ap_per_class[lbl] = 0.0
            continue

        # Sort by score descending (already sorted globally, but per-class may not be)
        entries.sort(key=lambda x: -x[0])
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        confs = []
        f1s = []

        for score, is_tp in entries:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
            r = tp_cum / n_gt
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            precisions.append(round(p, 4))
            recalls.append(round(r, 4))
            confs.append(round(score, 4))
            f1s.append(round(f1, 4))

        # AP: area under PR curve (all-point interpolation)
        ap = _compute_ap(precisions, recalls)
        ap_per_class[lbl] = round(ap, 4)

        # Downsample for frontend (max 200 points)
        step = max(1, len(precisions) // 200)
        pr_curves[lbl] = {
            "precision": precisions[::step],
            "recall": recalls[::step],
            "confidence": confs[::step],
        }
        f1_curves[lbl] = {
            "f1": f1s[::step],
            "confidence": confs[::step],
        }

    # mAP
    ap_values = [v for v in ap_per_class.values() if v > 0 or class_n_gt.get(list(ap_per_class.keys())[list(ap_per_class.values()).index(v)], 0) > 0]
    mAP = round(sum(ap_per_class.values()) / len(ap_per_class), 4) if ap_per_class else 0

    return {
        "labels": labels,
        "pr_curves": pr_curves,
        "f1_curves": f1_curves,
        "ap_per_class": ap_per_class,
        "mAP50": mAP,
        "confusion_matrix": conf_matrix,
        "total_gt": len(all_gt),
        "total_pred": len(all_pred),
    }


def _compute_ap(precisions, recalls):
    """Compute AP using all-point interpolation."""
    if not precisions:
        return 0.0
    # Add sentinel values
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Find points where recall changes
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Evaluate multiple models against GT, streaming progress via SSE."""
    data = request.get_json(force=True)
    gt_path = data.get("gt_path", "").strip()
    gt_format = data.get("gt_format", "auto")
    models = data.get("models", [])  # [{path, format, name}]
    iou_thresh = data.get("iou_thresh", 0.5)

    if not gt_path:
        return jsonify({"ok": False, "error": "GT 目录不能为空"}), 400
    gt_exp = os.path.expanduser(gt_path)
    if not os.path.isdir(gt_exp):
        return jsonify({"ok": False, "error": f"GT 目录不存在: {gt_exp}"}), 400
    if not models:
        return jsonify({"ok": False, "error": "至少需要一个模型"}), 400

    def generate():
        total = len(models)
        results = {}
        for idx, m in enumerate(models):
            m_path = os.path.expanduser(m.get("path", "").strip())
            m_fmt = m.get("format", "auto")
            m_name = m.get("name", os.path.basename(m_path))
            yield f"data: {json.dumps({'type':'progress','current':idx,'total':total,'name':m_name})}\n\n"
            if not os.path.isdir(m_path):
                results[m_name] = {"error": f"目录不存在: {m_path}"}
                continue
            try:
                ev = _evaluate_model(gt_exp, gt_format, m_path, m_fmt, iou_thresh)
                results[m_name] = ev
            except Exception as e:
                results[m_name] = {"error": str(e)}
        yield f"data: {json.dumps({'type':'done','ok':True,'results':results}, ensure_ascii=False)}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')


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


# ============ SAM3 AI-assisted labeling ============
import sys
import numpy as np
import cv2 as cv2_mod

_SAM3_DIR = os.path.join(os.path.dirname(__file__), "models")
_SAM3_LIB = os.path.join(
    os.path.dirname(__file__),
    "..", "X-AnyLabeling-Server", "app", "models"
)
_sam3_model = None
_sam3_processor = None
_sam3_state = None       # inference state for current image
_sam3_current_img = None  # name of image currently embedded


def _sam3_ensure_loaded():
    """Lazy-load SAM3 model on first use."""
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        return True

    # Add X-AnyLabeling-Server model code to path
    lib_path = os.path.abspath(_SAM3_LIB)
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"SAM3 依赖未安装: {e}")
        return False

    model_path = os.path.join(_SAM3_DIR, "sam3.pt")
    bpe_path = os.path.join(_SAM3_DIR, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(model_path):
        print(f"SAM3 模型文件不存在: {model_path}")
        return False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 SAM3 模型: {model_path} (device={device})")
    _sam3_model = build_sam3_image_model(
        bpe_path=bpe_path, device=device, checkpoint_path=model_path,
    )
    _sam3_processor = Sam3Processor(_sam3_model, device=device)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()

    print("SAM3 模型加载完成")
    return True


def _sam3_embed_image(img_name):
    """Embed image for SAM3 (cached per image name)."""
    global _sam3_state, _sam3_current_img
    if _sam3_current_img == img_name and _sam3_state is not None:
        return _sam3_state
    img_path = _img_dir() / img_name
    pil_img = Image.open(img_path).convert("RGB")
    _sam3_state = _sam3_processor.set_image(pil_img)
    _sam3_current_img = img_name
    return _sam3_state


def _mask_to_polygon(mask, epsilon_factor=0.001):
    """Convert binary mask to polygon points."""
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2_mod.findContours(mask_uint8, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2_mod.contourArea)
    if epsilon_factor > 0:
        eps = epsilon_factor * cv2_mod.arcLength(largest, True)
        approx = cv2_mod.approxPolyDP(largest, eps, True)
    else:
        approx = largest
    pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _sam3_results_to_shapes(results, labels_list=None, show_boxes=True, show_masks=False):
    """Convert SAM3 output to X-AnyLabeling shape dicts."""
    shapes = []
    if "scores" not in results or len(results["scores"]) == 0:
        return shapes
    import torch
    boxes = results["boxes"].cpu().float().numpy()
    scores = results["scores"].cpu().float().numpy()
    masks = results["masks"].cpu().float().numpy()
    labels = labels_list or []
    for i in range(len(scores)):
        lbl = labels[i] if i < len(labels) else "object"
        score = float(scores[i])
        if show_masks:
            pts = _mask_to_polygon(masks[i].squeeze())
            if pts:
                shapes.append({"label": lbl, "shape_type": "polygon", "points": pts, "score": score})
        if show_boxes:
            x1, y1, x2, y2 = boxes[i]
            shapes.append({
                "label": lbl, "shape_type": "rectangle",
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "score": score,
            })
    return shapes


@app.route("/api/sam3/status")
def api_sam3_status():
    """Check if SAM3 model is available."""
    model_path = os.path.join(_SAM3_DIR, "sam3.pt")
    loaded = _sam3_model is not None
    available = os.path.exists(model_path)
    return jsonify({"available": available, "loaded": loaded})


@app.route("/api/sam3/text", methods=["POST"])
def api_sam3_text():
    """Text prompt segmentation."""
    if not _sam3_ensure_loaded():
        return jsonify({"ok": False, "error": "SAM3 模型未加载"}), 500
    data = request.get_json(force=True)
    img_name = data.get("image", "")
    text = data.get("text", "").strip()
    conf = data.get("conf", 0.3)
    show_masks = data.get("show_masks", False)
    if not img_name or not text:
        return jsonify({"ok": False, "error": "缺少图片名或文本"}), 400

    state = _sam3_embed_image(img_name)
    _sam3_processor.set_confidence_threshold(conf)

    # Support comma/period separated multi-prompt
    sep = None
    for s in [",", ".", "，", "。"]:
        if s in text:
            sep = s
            break
    prompts = [p.strip() for p in (text.split(sep) if sep else [text]) if p.strip()]
    prompts = list(dict.fromkeys(prompts))  # deduplicate

    import torch
    all_masks, all_boxes, all_scores, all_labels = [], [], [], []
    for prompt in prompts:
        _sam3_processor.reset_all_prompts(state)
        out = _sam3_processor.set_text_prompt(state=state, prompt=prompt)
        n = len(out.get("scores", []))
        if n > 0:
            all_masks.append(out["masks"])
            all_boxes.append(out["boxes"])
            all_scores.append(out["scores"])
            all_labels.extend([prompt] * n)

    if all_masks:
        combined = {
            "masks": torch.cat(all_masks), "boxes": torch.cat(all_boxes),
            "scores": torch.cat(all_scores),
        }
        shapes = _sam3_results_to_shapes(combined, all_labels, show_boxes=True, show_masks=show_masks)
    else:
        shapes = []

    return jsonify({"ok": True, "shapes": shapes})


@app.route("/api/sam3/point", methods=["POST"])
def api_sam3_point():
    """Point prompt segmentation."""
    if not _sam3_ensure_loaded():
        return jsonify({"ok": False, "error": "SAM3 模型未加载"}), 500
    data = request.get_json(force=True)
    img_name = data.get("image", "")
    points = data.get("points", [])  # [{x, y, positive}]
    text = data.get("text", "").strip()
    conf = data.get("conf", 0.3)
    show_masks = data.get("show_masks", True)
    if not img_name or not points:
        return jsonify({"ok": False, "error": "缺少图片名或点"}), 400

    state = _sam3_embed_image(img_name)
    _sam3_processor.set_confidence_threshold(conf)
    _sam3_processor.reset_all_prompts(state)

    # If text provided, set it first for better results
    if text:
        _sam3_processor.set_text_prompt(state=state, prompt=text)

    # Get image size for normalization
    img_path = _img_dir() / img_name
    pil_img = Image.open(img_path)
    w, h = pil_img.size

    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import normalize_bbox
    import torch

    # Convert points to box prompts (small box around each point)
    for pt in points:
        px, py = pt["x"], pt["y"]
        positive = pt.get("positive", True)
        # Create a small box around the point
        sz = 2
        box_xywh = [px - sz, py - sz, sz * 2, sz * 2]
        box_cxcywh = box_xywh_to_cxcywh(
            torch.tensor(box_xywh, device=_sam3_processor.device).view(-1, 4)
        )
        norm_box = normalize_bbox(box_cxcywh, w, h).tolist()[0]
        state = _sam3_processor.add_geometric_prompt(
            state=state, box=norm_box, label=positive,
        )

    label = text or "object"
    n = len(state.get("scores", []))
    labels_list = [label] * n
    shapes = _sam3_results_to_shapes(state, labels_list, show_boxes=True, show_masks=show_masks)
    return jsonify({"ok": True, "shapes": shapes})


@app.route("/api/sam3/batch", methods=["POST"])
def api_sam3_batch():
    """Batch text prompt segmentation with SSE progress."""
    if not _sam3_ensure_loaded():
        return jsonify({"ok": False, "error": "SAM3 模型未加载"}), 500
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    images = data.get("images", [])  # list of image names
    conf = data.get("conf", 0.3)
    show_masks = data.get("show_masks", False)
    skip_existing = data.get("skip_existing", True)
    if not text or not images:
        return jsonify({"ok": False, "error": "缺少文本或图片列表"}), 400

    import torch

    # Parse prompts once
    sep = None
    for s in [",", ".", "，", "。"]:
        if s in text:
            sep = s
            break
    prompts = [p.strip() for p in (text.split(sep) if sep else [text]) if p.strip()]
    prompts = list(dict.fromkeys(prompts))

    def generate():
        total = len(images)
        done_count = 0
        total_objects = 0
        for i, img_name in enumerate(images):
            yield f"data: {json.dumps({'type':'progress','current':i,'total':total,'image':img_name})}\n\n"

            # Skip images that already have annotations
            if skip_existing:
                ann = _read_annotation(img_name)
                if ann.get("shapes"):
                    done_count += 1
                    continue

            try:
                state = _sam3_embed_image(img_name)
                _sam3_processor.set_confidence_threshold(conf)
                all_masks, all_boxes, all_scores, all_labels = [], [], [], []
                for prompt in prompts:
                    _sam3_processor.reset_all_prompts(state)
                    out = _sam3_processor.set_text_prompt(state=state, prompt=prompt)
                    n = len(out.get("scores", []))
                    if n > 0:
                        all_masks.append(out["masks"])
                        all_boxes.append(out["boxes"])
                        all_scores.append(out["scores"])
                        all_labels.extend([prompt] * n)

                if all_masks:
                    combined = {
                        "masks": torch.cat(all_masks), "boxes": torch.cat(all_boxes),
                        "scores": torch.cat(all_scores),
                    }
                    shapes = _sam3_results_to_shapes(combined, all_labels, show_boxes=True, show_masks=show_masks)
                else:
                    shapes = []

                total_objects += len(shapes)

                # Read existing annotation and merge/replace
                ann = _read_annotation(img_name)
                w, h = ann.get("imageWidth", 0), ann.get("imageHeight", 0)
                if w == 0 or h == 0:
                    sz = _get_img_size(img_name)
                    w, h = sz
                save_data = {
                    "version": "2.4.0", "flags": {},
                    "shapes": shapes,
                    "imagePath": img_name, "imageData": None,
                    "imageHeight": h, "imageWidth": w,
                }
                jp = _label_dir() / (Path(img_name).stem + ".json")
                jp.parent.mkdir(parents=True, exist_ok=True)
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                done_count += 1
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','image':img_name,'error':str(e)})}\n\n"

        global _label_cache
        _label_cache = None
        yield f"data: {json.dumps({'type':'done','total':total,'done':done_count,'objects':total_objects})}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')


# ============ Confirm sample (copy to verified dir) ============

_confirm_dir = None  # output directory for confirmed samples


@app.route("/api/confirm_path", methods=["GET"])
def api_get_confirm_path():
    return jsonify({"path": _confirm_dir or ""})


@app.route("/api/confirm_path", methods=["POST"])
def api_set_confirm_path():
    global _confirm_dir
    data = request.get_json(force=True)
    p = data.get("path", "").strip()
    if p:
        expanded = os.path.expanduser(p)
        os.makedirs(expanded, exist_ok=True)
        _confirm_dir = expanded
    else:
        _confirm_dir = None
    return jsonify({"ok": True, "path": _confirm_dir or ""})


@app.route("/api/confirm/<path:name>", methods=["POST"])
def api_confirm_sample(name):
    """Copy image + annotation to the confirmed output directory."""
    import shutil
    if not _confirm_dir:
        return jsonify({"ok": False, "error": "未设置确认输出目录"}), 400

    os.makedirs(_confirm_dir, exist_ok=True)
    img_labels_dir = os.path.join(_confirm_dir, "labels")
    img_images_dir = os.path.join(_confirm_dir, "images")
    os.makedirs(img_labels_dir, exist_ok=True)
    os.makedirs(img_images_dir, exist_ok=True)

    # Copy image
    src_img = _img_dir() / name
    if src_img.exists():
        shutil.copy2(str(src_img), os.path.join(img_images_dir, name))

    # Copy annotation in current format
    fmt = _get_current_format()
    stem = Path(name).stem
    if fmt == "yolo":
        src_ann = _label_dir() / (stem + ".txt")
        if src_ann.exists():
            shutil.copy2(str(src_ann), os.path.join(img_labels_dir, stem + ".txt"))
    elif fmt == "voc":
        src_ann = _label_dir() / (stem + ".xml")
        if src_ann.exists():
            shutil.copy2(str(src_ann), os.path.join(img_labels_dir, stem + ".xml"))
    else:
        src_ann = _label_dir() / (stem + ".json")
        if src_ann.exists():
            shutil.copy2(str(src_ann), os.path.join(img_labels_dir, stem + ".json"))

    return jsonify({"ok": True})


# ============ Classification tag definitions ============

_classify_tags_file = None  # resolved path, set in main() or lazily


def _get_classify_tags_file():
    global _classify_tags_file
    if _classify_tags_file:
        return Path(_classify_tags_file)
    return _label_dir() / ".classify_tags.json"


@app.route("/api/classify_tags", methods=["GET"])
def api_get_classify_tags():
    fp = _get_classify_tags_file()
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({"tags": []})


@app.route("/api/classify_tags", methods=["POST"])
def api_set_classify_tags():
    data = request.get_json(force=True)
    tags = data.get("tags", [])
    fp = _get_classify_tags_file()
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"tags": tags}, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True})


# ============ TensorBoard management ============
_tb_process = None   # subprocess.Popen instance
_tb_port = None      # port TensorBoard is running on
_tb_logdir = None    # current logdir


def _find_free_port():
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@app.route("/api/tensorboard/status")
def api_tb_status():
    global _tb_process
    running = _tb_process is not None and _tb_process.poll() is None
    if not running and _tb_process is not None:
        _tb_process = None
    return jsonify({
        "running": running,
        "port": _tb_port if running else None,
        "logdir": _tb_logdir if running else None,
    })


@app.route("/api/tensorboard/start", methods=["POST"])
def api_tb_start():
    global _tb_process, _tb_port, _tb_logdir
    import subprocess, shutil

    data = request.get_json(force=True)
    logdir = data.get("logdir", "").strip()
    if not logdir:
        return jsonify({"ok": False, "error": "日志目录不能为空"}), 400
    logdir = os.path.expanduser(logdir)
    if not os.path.isdir(logdir):
        return jsonify({"ok": False, "error": f"目录不存在: {logdir}"}), 400

    # Check if tensorboard is installed
    tb_bin = shutil.which("tensorboard")
    if not tb_bin:
        return jsonify({"ok": False, "error": "未找到 tensorboard 命令，请先安装: pip install tensorboard"}), 400

    # Stop existing instance if running
    if _tb_process and _tb_process.poll() is None:
        _tb_process.terminate()
        try:
            _tb_process.wait(timeout=5)
        except Exception:
            _tb_process.kill()

    port = _find_free_port()
    try:
        _tb_process = subprocess.Popen(
            [tb_bin, "--logdir", logdir, "--port", str(port), "--bind_all", "--reload_interval", "15"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _tb_port = port
        _tb_logdir = logdir
        return jsonify({"ok": True, "port": port, "logdir": logdir})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/tensorboard/stop", methods=["POST"])
def api_tb_stop():
    global _tb_process, _tb_port, _tb_logdir
    if _tb_process and _tb_process.poll() is None:
        _tb_process.terminate()
        try:
            _tb_process.wait(timeout=5)
        except Exception:
            _tb_process.kill()
    _tb_process = None
    _tb_port = None
    _tb_logdir = None
    return jsonify({"ok": True})


def main():
    import argparse
    import atexit
    global DATA_DIR, LABEL_DIR, LABEL_FORMAT

    def _cleanup_tb():
        global _tb_process
        if _tb_process and _tb_process.poll() is None:
            _tb_process.terminate()
    atexit.register(_cleanup_tb)

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


if __name__ == "__main__":
    main()
