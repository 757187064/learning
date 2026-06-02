#!/usr/bin/env python3
import hashlib
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "深度学习复习网站"
DATABASE = OUT / "database"
SITE = OUT / "site"
ASSETS = SITE / "assets"
IMAGE_OUT = SITE / "image"
MINDMAP_OUT = SITE / "mindmaps"
CACHE_VERSION = 3


KEY_TERMS = [
    "梯度下降", "损失函数", "学习率", "Epoch", "Batch", "反向传播", "计算图", "自动微分",
    "Logistic", "Sigmoid", "Softmax", "交叉熵", "BCEWithLogitsLoss", "CrossEntropyLoss",
    "MLP", "多层感知机", "激活函数", "仿射变换", "卷积", "互相关", "卷积核", "步长", "填充",
    "Valid", "Same", "Full", "感受野", "池化", "NCHW", "NHWC", "归一化", "标准化",
    "LeNet", "AlexNet", "ImageFolder", "WeightedRandomSampler", "类别不平衡",
    "梯度消失", "梯度爆炸", "梯度裁剪", "Xavier", "Kaiming", "BatchNorm", "Dropout",
    "数据增强", "权重衰减", "早停", "优化器", "Momentum", "Nesterov", "Adagrad", "RMSProp",
    "Adam", "学习率调度器", "ReduceLROnPlateau", "超参数", "消融实验",
    "VGG", "GoogLeNet", "Inception", "ResNet", "残差连接", "迁移学习",
    "RNN", "隐藏状态", "双向RNN", "堆叠RNN", "GRU", "LSTM", "Padding", "Packing",
    "PackedSequence", "1D卷积", "TCN", "Seq2Seq", "Encoder-Decoder", "Teacher Forcing",
    "注意力机制", "Attention", "Query", "Key", "Value", "Q/K/V", "上下文向量",
    "Scaled Dot-Product Attention", "sqrt(d_k)", "Source Mask", "Multi-Head Attention",
    "Self-Attention", "Cross-Attention", "Target Mask", "Subsequent Mask",
    "位置编码", "Positional Encoding", "Transformer", "RoPE",
]

BAD_QUESTION_PATTERNS = [
    "核心词是",
    "这句话描述的核心词",
    "必须抓住的两个要点",
    "围绕“",
    "围绕\"",
    "最适合作为考试标准答案",
]


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def compact(text: str, max_len: int = 260) -> str:
    text = re.sub(r"\s+", " ", strip_html(text))
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("，。；:： ") + "..."


def heading_level(line: str):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not match:
        return None
    title = strip_html(match.group(2)).strip(" #")
    if not title:
        return None
    return len(match.group(1)), title


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def terms_in(text: str):
    return [term for term in KEY_TERMS if re.search(re.escape(term), text, flags=re.I)]


def load_cache():
    path = DATABASE / "source_cache.json"
    if not path.exists():
        return {"version": CACHE_VERSION, "files": {}}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": CACHE_VERSION, "files": {}}
    if cache.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "files": {}}
    cache.setdefault("files", {})
    return cache


def extract_notebook(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    manifest = {
        "file": path.name,
        "kind": "ipynb",
        "size_bytes": path.stat().st_size,
        "cell_count": len(cells),
        "markdown_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
        "code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
    }
    chunks = []
    path_stack = []
    for i, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            for raw_line in source.splitlines():
                parsed = heading_level(raw_line)
                if parsed:
                    level, title = parsed
                    path_stack = path_stack[: level - 1]
                    path_stack.append(title)
        clean = strip_html(source)
        if not clean:
            continue
        title = path_stack[-1] if path_stack else path.stem
        chunks.append({
            "id": f"{path.stem}::cell-{i + 1}",
            "file": path.name,
            "kind": "ipynb",
            "cell_index": i + 1,
            "slide_index": None,
            "type": cell.get("cell_type", "unknown"),
            "title": title,
            "heading_path": path_stack[-4:],
            "text": clean,
            "summary": compact(clean, 680),
            "terms": terms_in(clean),
        })
    return manifest, chunks


def extract_pptx(path: Path):
    try:
        from pptx import Presentation
    except Exception:
        manifest = {
            "file": path.name,
            "kind": "pptx",
            "size_bytes": path.stat().st_size,
            "slide_count": 0,
            "warning": "python-pptx unavailable; skipped extraction",
        }
        return manifest, []

    prs = Presentation(str(path))
    manifest = {
        "file": path.name,
        "kind": "pptx",
        "size_bytes": path.stat().st_size,
        "slide_count": len(prs.slides),
    }
    chunks = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        title = f"第 {i} 页"
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            value = strip_html(shape.text)
            if not value:
                continue
            if title == f"第 {i} 页":
                title = value.splitlines()[0][:80]
            texts.append(value)
        try:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                texts.append("备注：" + strip_html(notes))
        except Exception:
            pass
        clean = strip_html("\n".join(texts))
        if not clean:
            continue
        chunks.append({
            "id": f"{path.stem}::slide-{i}",
            "file": path.name,
            "kind": "pptx",
            "cell_index": None,
            "slide_index": i,
            "type": "slide",
            "title": title,
            "heading_path": [title],
            "text": clean,
            "summary": compact(clean, 680),
            "terms": terms_in(clean),
        })
    return manifest, chunks


def extract_sources():
    cache = load_cache()
    manifest = []
    chunks = []
    files = sorted(list(ROOT.glob("*.ipynb")) + list(ROOT.glob("*.pptx")))
    new_cache = {"version": CACHE_VERSION, "files": {}}
    for path in files:
        sha = file_hash(path)
        cached = cache.get("files", {}).get(path.name)
        if cached and cached.get("sha256") == sha:
            item_manifest = cached["manifest"]
            item_chunks = cached["chunks"]
        else:
            if path.suffix.lower() == ".pptx":
                item_manifest, item_chunks = extract_pptx(path)
            else:
                item_manifest, item_chunks = extract_notebook(path)
            item_manifest["sha256"] = sha
        item_manifest["sha256"] = sha
        manifest.append(item_manifest)
        chunks.extend(item_chunks)
        new_cache["files"][path.name] = {
            "sha256": sha,
            "manifest": item_manifest,
            "chunks": item_chunks,
        }
    return manifest, chunks, new_cache


def build_term_index(chunks):
    term_data = {}
    for term in KEY_TERMS:
        hits = []
        for chunk in chunks:
            count = len(re.findall(re.escape(term), chunk["text"], flags=re.I))
            if count:
                hits.append({
                    "chunk_id": chunk["id"],
                    "file": chunk["file"],
                    "kind": chunk.get("kind"),
                    "cell_index": chunk.get("cell_index"),
                    "slide_index": chunk.get("slide_index"),
                    "title": chunk["title"],
                    "count": count,
                    "summary": chunk["summary"],
                })
        if hits:
            related = Counter()
            by_id = {c["id"]: c for c in chunks}
            for hit in hits:
                related.update(t for t in by_id[hit["chunk_id"]]["terms"] if t != term)
            term_data[term] = {
                "term": term,
                "total_count": sum(h["count"] for h in hits),
                "files": sorted({h["file"] for h in hits}),
                "hits": hits[:80],
                "related": [t for t, _ in related.most_common(8)],
            }
    return term_data


def choose_chapter(chunk):
    path = [p for p in chunk.get("heading_path", []) if p]
    if not path:
        return chunk.get("title") or chunk["file"]
    section_re = re.compile(r"(第\s*\d+\s*[章节]|第[一二三四五六七八九十]+[章节])")
    section_indices = [i for i, title in enumerate(path) if section_re.search(title)]
    if section_indices:
        idx = section_indices[-1]
        if idx > 0 and "章" in path[idx - 1]:
            return f"{path[idx - 1]} / {path[idx]}"
        return path[idx]
    if len(path) >= 2 and "章" in path[0]:
        return f"{path[0]} / {path[1]}"
    return path[0]


def is_structural_chapter(title):
    return bool(re.search(r"(第\s*\d+\s*[章节]|第[一二三四五六七八九十]+[章节]|课堂练习)", title))


def concept_lookup_from_quiz(existing_quiz):
    lookup = {}
    for q in existing_quiz:
        topic = q.get("topic", "")
        stem = q.get("stem", "")
        text = " ".join([stem, q.get("answer", ""), q.get("explanation", "")])
        for term in KEY_TERMS:
            if term in text or term in topic:
                item = lookup.setdefault(term, {"tips": [], "pitfalls": []})
                explanation = q.get("explanation", "")
                if explanation:
                    item["tips"].append(compact(explanation, 160))
                if "错误" in explanation:
                    item["pitfalls"].append(compact(explanation, 140))
    return lookup


def source_label(file_name, chunk):
    if chunk.get("slide_index"):
        return f"{file_name} · 第 {chunk['slide_index']} 页"
    return f"{file_name} · cell {chunk.get('cell_index')}"


def build_course_outline(manifest, chunks, quiz_lookup):
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk["file"]].append(chunk)

    courses = []
    for source in manifest:
        file_name = source["file"]
        source_chunks = by_file.get(file_name, [])
        course_chapter = ""
        for chunk in source_chunks:
            for part in chunk.get("heading_path", []):
                if "章" in part:
                    course_chapter = part
                    break
            if course_chapter:
                break
        chapter_map = {}
        order = []
        current_chapter = ""
        for chunk in source_chunks:
            candidate = choose_chapter(chunk)
            if course_chapter and re.match(r"^第\s*\d+\s*节|^第[一二三四五六七八九十]+节", candidate):
                candidate = f"{course_chapter} / {candidate}"
            if not is_structural_chapter(candidate) and current_chapter:
                chapter = current_chapter
            else:
                chapter = candidate
            if is_structural_chapter(chapter):
                current_chapter = chapter
            chapter = chapter or Path(file_name).stem
            if chapter not in chapter_map:
                chapter_map[chapter] = {"title": chapter, "chunks": [], "terms": Counter()}
                order.append(chapter)
            chapter_map[chapter]["chunks"].append(chunk)
            chapter_map[chapter]["terms"].update(chunk.get("terms") or [])

        chapters = []
        for chapter_name in order:
            chapter = chapter_map[chapter_name]
            rows = []
            ranked_terms = [term for term, _ in chapter["terms"].most_common()]
            if not ranked_terms:
                ranked_terms = [chapter_name]
            for term in ranked_terms[:18]:
                term_chunks = [c for c in chapter["chunks"] if term in c.get("terms", [])] or chapter["chunks"][:2]
                examples = " ".join(c["summary"] for c in term_chunks[:2])
                quiz_info = quiz_lookup.get(term, {})
                tip = quiz_info.get("tips", [compact(examples, 130)])[0]
                pitfall = quiz_info.get("pitfalls", ["注意区分它的作用、输入输出位置和训练/推理阶段。"])[0]
                rows.append({
                    "term": term,
                    "plain": compact(examples, 150),
                    "exam": tip,
                    "pitfall": pitfall,
                    "memory": compact(examples, 110),
                    "source": "；".join(source_label(file_name, c) for c in term_chunks[:3]),
                })
            chapters.append({
                "title": chapter_name,
                "chunk_count": len(chapter["chunks"]),
                "terms": ranked_terms[:24],
                "rows": rows,
            })
        courses.append({
            "file": file_name,
            "kind": source.get("kind"),
            "title": Path(file_name).stem,
            "chapter_count": len(chapters),
            "chunk_count": len(source_chunks),
            "chapters": chapters,
        })

    return {
        "title": "按课件整理的期末速记目录",
        "courses": courses,
    }


def safe_slug(text):
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text).strip("-")
    return slug[:80] or "mindmap"


def svg_text(text, x, y, size=14, weight="400", fill="#172b4d"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(compact(text, 42))}</text>'


def write_mindmap_svg(path, title, chapters):
    rows = []
    for chapter in chapters:
        terms = chapter.get("terms") or [r["term"] for r in chapter.get("rows", [])]
        rows.append((chapter["title"], terms[:14]))
    height = max(260, 120 + sum(max(1, len(t)) * 34 + 34 for _, t in rows))
    width = 1180
    y = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}.node{fill:#fff;stroke:#d8dee8;stroke-width:1.2}.root{fill:#e8f4ff;stroke:#1a73e8}.chapter{fill:#f7f9fc}.term{fill:#fff}</style>',
        f'<rect class="node root" x="28" y="28" rx="12" width="240" height="54"/>',
        svg_text(title, 46, 61, 16, "700", "#174ea6"),
    ]
    for chapter_title, terms in rows:
        chapter_y = y
        parts.append(f'<path d="M268 {chapter_y + 20} C330 {chapter_y + 20}, 330 {chapter_y + 20}, 382 {chapter_y + 20}" stroke="#b6c2d4" fill="none"/>')
        parts.append(f'<rect class="node chapter" x="382" y="{chapter_y}" rx="10" width="250" height="42"/>')
        parts.append(svg_text(chapter_title, 400, chapter_y + 26, 14, "700", "#263238"))
        term_y = chapter_y
        for term in terms:
            parts.append(f'<path d="M632 {chapter_y + 20} C680 {chapter_y + 20}, 680 {term_y + 20}, 722 {term_y + 20}" stroke="#d0d7e2" fill="none"/>')
            parts.append(f'<rect class="node term" x="722" y="{term_y}" rx="10" width="320" height="40"/>')
            parts.append(svg_text(term, 740, term_y + 25, 13, "600", "#172b4d"))
            term_y += 34
        y += max(1, len(terms)) * 34 + 48
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_mindmaps(outline):
    MINDMAP_OUT.mkdir(parents=True, exist_ok=True)
    for old in MINDMAP_OUT.glob("*.svg"):
        old.unlink()
    maps = []
    for course in outline["courses"]:
        chapters = course["chapters"]
        total_terms = sum(len(ch.get("terms", [])) for ch in chapters)
        course_maps = []
        if total_terms > 42 or len(chapters) > 8:
            for idx, chapter in enumerate(chapters, 1):
                file_name = f"{safe_slug(course['title'])}-{idx:02d}-{safe_slug(chapter['title'])}.svg"
                rel = f"mindmaps/{file_name}"
                write_mindmap_svg(SITE / rel, f"{course['title']} / {chapter['title']}", [chapter])
                course_maps.append({
                    "title": chapter["title"],
                    "node_count": 1 + len(chapter.get("terms", [])),
                    "svg": rel,
                    "chapters": [chapter],
                })
        else:
            file_name = f"{safe_slug(course['title'])}.svg"
            rel = f"mindmaps/{file_name}"
            write_mindmap_svg(SITE / rel, course["title"], chapters)
            course_maps.append({
                "title": "整份课件",
                "node_count": 1 + len(chapters) + total_terms,
                "svg": rel,
                "chapters": chapters,
            })
        maps.append({
            "file": course["file"],
            "title": course["title"],
            "kind": course.get("kind"),
            "maps": course_maps,
        })
    return maps


def q_mc(id_, topic, stem, options, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "mcq", "topic": topic, "stem": stem, "options": options, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_fill(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "fill", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_tf(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "tf", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_short(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "short", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def has_bad_question_text(q):
    text = " ".join(str(q.get(k, "")) for k in ["stem", "answer", "explanation"])
    return any(p in text for p in BAD_QUESTION_PATTERNS)


def load_standard_quiz():
    path = DATABASE / "quiz_bank.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []
    seen = set()
    for q in data:
        if has_bad_question_text(q):
            continue
        qid = q.get("id")
        if not qid or qid in seen:
            continue
        cleaned.append(q)
        seen.add(qid)
    return cleaned


def build_beginner_quiz(outline):
    questions = []
    idx = 1
    for course in outline["courses"]:
        for chapter in course["chapters"]:
            for row in chapter["rows"][:10]:
                term = row["term"]
                source = course["file"]
                topic = f"{course['title']} / {chapter['title']}"
                plain = row["plain"] or row["memory"]
                if not plain:
                    continue
                questions.append(q_mc(
                    f"BMC{idx:04d}",
                    topic,
                    f"在本章知识脉络中，{term}最主要对应下面哪一类内容？",
                    [
                        f"A. {compact(plain, 92)}",
                        "B. 只用于装饰课件标题，不参与模型或训练理解",
                        "C. 只在测试集上修改标签，训练阶段不需要关注",
                        "D. 与输入、模型、损失或评估都没有关系",
                    ],
                    "A",
                    f"A正确。{compact(row['exam'], 150)}",
                    source,
                ))
                questions.append(q_tf(
                    f"BTF{idx:04d}",
                    topic,
                    f"{term}复习时应结合它所在的模型结构、训练流程、输入输出形状或公式条件来判断，而不是只背一个孤立名词。",
                    "正确",
                    "基础阶段先把知识点放回课程脉络中，后面再做标准题会更稳。",
                    source,
                ))
                questions.append(q_fill(
                    f"BF{idx:04d}",
                    topic,
                    f"本章中与“{compact(plain, 58)}”关系最直接的知识点是____。",
                    term,
                    f"答案是{term}。{compact(row['memory'], 150)}",
                    source,
                ))
                if idx % 3 == 0:
                    questions.append(q_short(
                        f"BS{idx:04d}",
                        topic,
                        f"用两句话说明{term}在本章中的作用，并写出一个复习时容易忽略的点。",
                        f"{term}的作用可以概括为：{compact(plain, 160)} 容易忽略的是：{compact(row['pitfall'], 130)}",
                        "基础简答不追求展开太深，先讲清“它是什么、放在哪里、容易错在哪里”。",
                        source,
                    ))
                idx += 1
    return [q for q in questions if not has_bad_question_text(q)]


def fallback_review(outline):
    return {
        "title": "深度学习课件综合复习资料",
        "updated_from": "当前课件目录内全部 Jupyter/PPT 课件",
        "sections": [
            {
                "title": course["title"],
                "points": [
                    f"本课件包含 {course['chapter_count']} 个章节块、{course['chunk_count']} 个可检索片段。",
                    "建议先看本页教材目录表格，再进入思维导图按知识脉络回忆，最后做基础练习和标准组卷。",
                ],
                "exam_focus": [row["term"] for ch in course["chapters"] for row in ch["rows"][:3]][:12],
            }
            for course in outline["courses"]
        ],
    }


def load_review_or_fallback(outline):
    path = DATABASE / "review_material.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["updated_from"] = "当前课件目录内全部 Jupyter/PPT 课件"
            return data
        except json.JSONDecodeError:
            pass
    return fallback_review(outline)


CSS = r"""
:root {
  --blue: #20a8e0;
  --blue-dark: #1376a8;
  --ink: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --soft: #f6f8fb;
  --panel: #ffffff;
  --green: #16a34a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--ink);
  background: #f7f9fc;
  letter-spacing: 0;
}
a { color: var(--blue-dark); text-decoration: none; }
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(255,255,255,.96);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(8px);
}
.topbar {
  max-width: 1440px;
  margin: 0 auto;
  padding: 14px 24px;
  display: grid;
  grid-template-columns: 290px 1fr;
  gap: 24px;
  align-items: center;
}
.brand h1 { margin: 0; font-size: 20px; line-height: 1.2; }
.brand p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
nav {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
nav button {
  border: 0;
  background: transparent;
  color: #445166;
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
}
nav button.active { color: #fff; background: var(--blue); }
main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 22px 24px 48px;
}
.view { display: none; }
.view.active { display: block; }
.hero-grid {
  display: grid;
  grid-template-columns: 1.15fr .85fr;
  gap: 18px;
  align-items: start;
  margin-bottom: 18px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
}
.panel h2, .panel h3 { margin: 0 0 12px; }
.muted, .meta { color: var(--muted); font-size: 13px; }
.stats {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
}
.stat {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.stat strong { display: block; font-size: 23px; color: #0f172a; }
.toolbar {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto auto;
  gap: 10px;
  margin: 14px 0;
}
input[type="search"] {
  border: 1px solid #d6dce6;
  border-radius: 8px;
  padding: 11px 13px;
  font-size: 15px;
  background: #fff;
}
.btn, .btn.secondary {
  border: 1px solid var(--blue);
  border-radius: 8px;
  padding: 10px 13px;
  cursor: pointer;
  font-weight: 700;
  background: var(--blue);
  color: #fff;
}
.btn.secondary { background: #fff; color: var(--blue-dark); }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 16px;
}
.result, .q, .answer-item {
  border-top: 1px solid var(--line);
  padding: 14px 0;
}
.result:first-child, .q:first-child, .answer-item:first-child { border-top: 0; }
.result-title {
  display: flex;
  gap: 10px;
  justify-content: space-between;
  align-items: flex-start;
  font-weight: 700;
}
.path { margin-top: 4px; color: var(--blue-dark); font-size: 13px; }
.snippet { margin: 8px 0 0; line-height: 1.65; }
mark { background: #dff3ff; color: #0b5e8e; border-radius: 3px; padding: 0 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
  border: 1px solid #dce3ec;
  background: #fff;
  color: #334155;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 13px;
  cursor: pointer;
}
.term-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: 8px;
}
.term-btn {
  border: 1px solid var(--line);
  background: #fff;
  text-align: left;
  border-radius: 8px;
  padding: 9px 10px;
  cursor: pointer;
}
.term-btn strong { display: block; }
.term-btn span { color: var(--muted); font-size: 12px; }
.course-layout {
  display: grid;
  grid-template-columns: 270px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.side-index {
  position: sticky;
  top: 92px;
  max-height: calc(100vh - 110px);
  overflow: auto;
}
.side-index a {
  display: block;
  padding: 8px 10px;
  border-radius: 8px;
  color: #334155;
  font-size: 13px;
}
.side-index a:hover { background: #edf7fd; color: var(--blue-dark); }
.course-card { margin-bottom: 16px; }
.course-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 12px;
}
.chapter { border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 11px;
  vertical-align: top;
  line-height: 1.55;
}
th { background: #f8fafc; text-align: left; color: #475569; font-size: 13px; }
tr:last-child td { border-bottom: 0; }
.quiz-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.quiz-section { margin: 18px 0; }
.q-stem { font-weight: 700; line-height: 1.6; }
.options { list-style: none; padding: 0; margin: 8px 0 0; }
.options li { margin: 5px 0; line-height: 1.5; }
.badge {
  display: inline-block;
  border: 1px solid #dbe3ec;
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--muted);
  font-size: 12px;
  margin-left: 6px;
}
.blank-answer {
  display: inline-block;
  min-width: 140px;
  border-bottom: 1px solid #9aa6b2;
}
.answers { background: #f8fbff; }
.mindmap-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
}
.map-card h3 { margin-bottom: 4px; }
.map-preview {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
  overflow: auto;
  max-height: 360px;
}
.map-tree ul { list-style: none; padding-left: 18px; border-left: 1px solid #dce6f2; }
.map-tree li { margin: 7px 0; line-height: 1.45; }
.map-node {
  display: inline-block;
  background: #fff;
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  padding: 5px 8px;
}
.map-node.term { color: #174ea6; cursor: pointer; }
.formula {
  font-family: "SFMono-Regular", Consolas, monospace;
  background: #f1f5f9;
  color: #0f4b6e;
  padding: 8px 10px;
  border-radius: 8px;
  margin: 6px 0;
  overflow-wrap: anywhere;
}
.review-section { border-top: 1px solid var(--line); padding-top: 18px; margin-top: 18px; }
@media (max-width: 920px) {
  .topbar, .hero-grid, .layout, .course-layout, .toolbar { grid-template-columns: 1fr; }
  .stats { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .side-index { position: static; max-height: none; }
}
"""


JS = r"""
const DB = window.COURSE_DB;
const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
}

function highlight(text, query) {
  const safe = escapeHtml(text || "");
  const q = (query || "").trim();
  if (!q) return safe;
  const parts = q.split(/\s+/).filter(Boolean).slice(0, 6).map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return safe;
  return safe.replace(new RegExp(`(${parts.join("|")})`, "gi"), "<mark>$1</mark>");
}

function scoreChunk(chunk, query) {
  const q = norm(query).split(/\s+/).filter(Boolean);
  if (!q.length) return 0;
  const title = norm(chunk.title + " " + (chunk.heading_path || []).join(" "));
  const text = norm(chunk.text);
  const terms = norm((chunk.terms || []).join(" "));
  let score = 0;
  q.forEach(part => {
    if (title.includes(part)) score += 10;
    if (terms.includes(part)) score += 8;
    const matches = text.match(new RegExp(part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"));
    if (matches) score += Math.min(matches.length, 12);
  });
  return score;
}

function shuffle(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function renderStats() {
  $("stats").innerHTML = `
    <div class="stat"><strong>${DB.manifest.length}</strong><span>课件文件</span></div>
    <div class="stat"><strong>${DB.chunks.length}</strong><span>可检索片段</span></div>
    <div class="stat"><strong>${Object.keys(DB.terms).length}</strong><span>索引词条</span></div>
    <div class="stat"><strong>${DB.quizBank.length}</strong><span>标准题</span></div>
    <div class="stat"><strong>${DB.beginnerQuizBank.length}</strong><span>基础题</span></div>
  `;
}

function renderTermCloud() {
  const terms = Object.values(DB.terms).sort((a, b) => b.total_count - a.total_count).slice(0, 42);
  $("termCloud").innerHTML = terms.map(t => `
    <button class="term-btn" data-term="${escapeHtml(t.term)}">
      <strong>${escapeHtml(t.term)}</strong>
      <span>${t.total_count} 次 · ${t.files.length} 文件</span>
    </button>
  `).join("");
  $("termCloud").querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    $("searchInput").value = btn.dataset.term;
    runSearch(btn.dataset.term);
    showView("search");
  }));
}

function runSearch(query) {
  const scored = DB.chunks.map(c => [scoreChunk(c, query), c]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 80);
  const results = scored.map(([, c]) => c);
  $("resultCount").textContent = query.trim() ? `${results.length} 个结果` : "等待输入";
  $("results").innerHTML = results.length ? results.map(c => `
    <article class="result">
      <div class="result-title">
        <span>${escapeHtml(c.title || c.file)}</span>
        <button class="chip result-quiz" data-topic="${escapeHtml((c.terms && c.terms[0]) || query)}">出题</button>
      </div>
      <div class="path">${escapeHtml((c.heading_path || []).join(" / "))}</div>
      <p class="snippet">${highlight(c.summary, query)}</p>
      <span class="meta">${escapeHtml(c.file)} · ${c.slide_index ? `第 ${c.slide_index} 页` : `cell ${c.cell_index}`} · ${escapeHtml(c.kind || c.type)}</span>
      <div class="chips">${(c.terms || []).slice(0, 8).map(t => `<button class="chip" data-term="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("")}</div>
    </article>
  `).join("") : `<div class="panel"><p class="meta">输入课件中的词条，例如“感受野”“BatchNorm”“LSTM”“CrossEntropyLoss”。</p></div>`;
  $("results").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
  $("results").querySelectorAll(".result-quiz").forEach(btn => btn.addEventListener("click", () => generateStandardQuiz(btn.dataset.topic)));

  const fuzzy = DB.terms[query] || Object.values(DB.terms).find(t => query && (norm(t.term).includes(norm(query)) || norm(query).includes(norm(t.term))));
  $("explainBox").innerHTML = fuzzy ? `
    <h2>${escapeHtml(fuzzy.term)}</h2>
    <p class="muted">出现 ${fuzzy.total_count} 次，分布在 ${fuzzy.files.length} 个课件文件中。</p>
    <div class="chips">${(fuzzy.related || []).map(r => `<button class="chip" data-term="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join("")}</div>
  ` : `<h2>搜索说明</h2><p class="muted">搜索会定位到课件原文片段、相关词条和来源位置。适合先查概念，再回到教材目录和导图复习。</p>`;
  $("explainBox").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
}

function renderOutline() {
  const courses = DB.outline.courses;
  $("courseIndex").innerHTML = courses.map((c, i) => `<a href="#course-${i}">${escapeHtml(c.title)}</a>`).join("");
  $("outlineBody").innerHTML = courses.map((course, i) => `
    <section class="panel course-card" id="course-${i}">
      <div class="course-head">
        <div>
          <h2>${escapeHtml(course.title)}</h2>
          <p class="muted">${escapeHtml(course.file)} · ${course.chapter_count} 个章节块 · ${course.chunk_count} 个片段</p>
        </div>
        <button class="btn secondary" data-map="${escapeHtml(course.file)}">看导图</button>
      </div>
      ${course.chapters.map(ch => `
        <div class="chapter">
          <h3>${escapeHtml(ch.title)}</h3>
          <table>
            <thead><tr><th>知识点</th><th>通俗解释</th><th>考试怎么考</th><th>易错点</th><th>必记句子/公式</th><th>来源</th></tr></thead>
            <tbody>${ch.rows.map(r => `
              <tr>
                <td><strong>${escapeHtml(r.term)}</strong></td>
                <td>${escapeHtml(r.plain)}</td>
                <td>${escapeHtml(r.exam)}</td>
                <td>${escapeHtml(r.pitfall)}</td>
                <td>${escapeHtml(r.memory)}</td>
                <td class="muted">${escapeHtml(r.source)}</td>
              </tr>
            `).join("")}</tbody>
          </table>
        </div>
      `).join("")}
    </section>
  `).join("");
  $("outlineBody").querySelectorAll("button[data-map]").forEach(btn => btn.addEventListener("click", () => {
    showView("mindmaps");
    const card = document.querySelector(`[data-map-card="${CSS.escape(btn.dataset.map)}"]`);
    if (card) card.scrollIntoView({behavior: "smooth", block: "start"});
  }));
}

function renderMapTree(map) {
  return `<div class="map-tree"><ul>${map.chapters.map(ch => `
    <li><span class="map-node">${escapeHtml(ch.title)}</span>
      <ul>${(ch.terms || []).slice(0, 16).map(t => `<li><span class="map-node term" data-term="${escapeHtml(t)}">${escapeHtml(t)}</span></li>`).join("")}</ul>
    </li>
  `).join("")}</ul></div>`;
}

function renderMindmaps() {
  $("mindmapBody").innerHTML = DB.mindmaps.map(course => `
    <article class="panel map-card" data-map-card="${escapeHtml(course.file)}">
      <h3>${escapeHtml(course.title)}</h3>
      <p class="muted">${escapeHtml(course.file)} · ${course.maps.length} 张导图${course.maps.length > 1 ? "，已按章节拆分" : ""}</p>
      ${course.maps.map(map => `
        <div class="chapter">
          <div class="result-title">
            <strong>${escapeHtml(map.title)} <span class="badge">${map.node_count} 节点</span></strong>
            <a class="btn secondary" href="${escapeHtml(map.svg)}" download>下载图片版</a>
          </div>
          <div class="map-preview">${renderMapTree(map)}</div>
        </div>
      `).join("")}
    </article>
  `).join("");
  $("mindmapBody").querySelectorAll(".map-node.term").forEach(node => node.addEventListener("click", () => {
    $("searchInput").value = node.dataset.term;
    runSearch(node.dataset.term);
    showView("search");
  }));
}

function pickQuestions(bank, type, topic, count) {
  const q = norm(topic || "");
  const all = bank.filter(x => x.type === type);
  const related = q ? all.filter(x => norm([x.topic, x.stem, x.source].join(" ")).includes(q)) : [];
  const base = related.length >= Math.min(count, 3) ? related : all;
  return shuffle(base).slice(0, count);
}

function generateQuiz(bank, mode, topic = "") {
  const spec = mode === "beginner"
    ? [["mcq", 10], ["tf", 10], ["fill", 10], ["short", 3]]
    : [["mcq", 20], ["fill", 10], ["short", 5]];
  const labels = {mcq: "单选题", tf: "判断题", fill: "填空题", short: "简答题"};
  const chosen = spec.flatMap(([type, count]) => pickQuestions(bank, type, topic, count));
  const title = mode === "beginner" ? "基础练习随机题" : "标准综合随机题";
  const targetTitle = mode === "beginner" ? $("beginnerTitle") : $("quizTitle");
  const targetMeta = mode === "beginner" ? $("beginnerMeta") : $("quizMeta");
  const targetBody = mode === "beginner" ? $("beginnerBody") : $("quizBody");
  targetTitle.textContent = topic ? `${title}：${topic}` : title;
  targetMeta.textContent = spec.map(([type, count]) => `${count} 道${labels[type]}`).join("、") + "。再次点击会重新随机生成。";
  targetBody.innerHTML = spec.map(([type]) => {
    const items = chosen.filter(q => q.type === type);
    return `<section class="quiz-section"><h3>${labels[type]}</h3>${items.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>`;
  }).join("") + `<section class="panel answers"><h3>答案与解析</h3>${chosen.map((q, i) => renderAnswer(q, i + 1)).join("")}</section>`;
}

function generateStandardQuiz(topic = "", activate = true) {
  generateQuiz(DB.quizBank, "standard", topic);
  if (activate) {
    showView("quiz");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function generateBeginnerQuiz(topic = "", activate = true) {
  generateQuiz(DB.beginnerQuizBank, "beginner", topic);
  if (activate) {
    showView("beginner");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function renderQuestion(q, idx) {
  if (q.type === "mcq") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options">${q.options.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
  }
  if (q.type === "tf") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options"><li>A. 正确</li><li>B. 错误</li></ul></div>`;
  }
  if (q.type === "fill") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="blank-answer"></span> <span class="badge">${escapeHtml(q.difficulty)}</span></div></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="muted">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="muted">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function renderReview() {
  const r = DB.review;
  $("reviewBody").innerHTML = (r.sections || []).map(sec => `
    <section class="review-section">
      <h2>${escapeHtml(sec.title)}</h2>
      <ul>${(sec.points || []).map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
      ${(sec.details || []).length ? `<h3>详细知识点</h3><ul>${sec.details.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.must_memorize || []).length ? `<h3>必背结论</h3><ul>${sec.must_memorize.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.formulas || []).map(f => `<div class="formula">${escapeHtml(f)}</div>`).join("")}
      ${(sec.exam_focus || []).length ? `<p><strong>常考点：</strong>${escapeHtml(sec.exam_focus.join("；"))}</p>` : ""}
    </section>
  `).join("");
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", e => runSearch(e.target.value));
  $("makeStandardQuiz").addEventListener("click", () => generateStandardQuiz($("searchInput").value.trim()));
  $("makeBeginnerQuiz").addEventListener("click", () => generateBeginnerQuiz($("searchInput").value.trim()));
  $("quizAll").addEventListener("click", () => generateStandardQuiz(""));
  $("beginnerAll").addEventListener("click", () => generateBeginnerQuiz(""));
  renderStats();
  renderTermCloud();
  renderOutline();
  renderMindmaps();
  renderReview();
  runSearch("");
  generateBeginnerQuiz("", false);
}

init();
"""


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>深度学习期末复习资料库</title>
  <link rel="stylesheet" href="assets/course_site.css">
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>深度学习期末复习资料库</h1>
        <p>检索 · 教材目录 · 思维导图 · 基础练习 · 标准组卷</p>
      </div>
      <nav aria-label="主导航">
        <button class="active" data-view="search">搜索</button>
        <button data-view="outline">教材目录</button>
        <button data-view="mindmaps">思维导图</button>
        <button data-view="beginner">基础练习</button>
        <button data-view="quiz">标准组卷</button>
        <button data-view="review">综合资料</button>
      </nav>
    </div>
  </header>
  <main>
    <section id="view-search" class="view active">
      <div class="hero-grid">
        <div class="panel">
          <h2>从一个词条开始复习</h2>
          <p class="muted">搜索会回到课件原文位置，也能继续跳到基础题、标准题、目录和导图。</p>
          <div class="toolbar">
            <input id="searchInput" type="search" placeholder="输入词条：如 感受野、BatchNorm、LSTM、CrossEntropyLoss">
            <button id="makeBeginnerQuiz" class="btn secondary">基础练习</button>
            <button id="makeStandardQuiz" class="btn">标准组卷</button>
          </div>
        </div>
        <div id="stats" class="stats"></div>
      </div>
      <div class="layout">
        <div>
          <div class="panel" style="margin-bottom:14px">
            <div class="result-title"><h2 style="margin:0">搜索结果</h2><span id="resultCount" class="meta"></span></div>
          </div>
          <div id="results" class="panel"></div>
        </div>
        <aside>
          <div id="explainBox" class="panel"></div>
          <div class="panel" style="margin-top:14px">
            <h3>高频词条</h3>
            <div id="termCloud" class="term-list"></div>
          </div>
        </aside>
      </div>
    </section>

    <section id="view-outline" class="view">
      <div class="course-layout">
        <aside class="panel side-index">
          <h3>课件目录</h3>
          <div id="courseIndex"></div>
        </aside>
        <div id="outlineBody"></div>
      </div>
    </section>

    <section id="view-mindmaps" class="view">
      <div class="panel" style="margin-bottom:14px">
        <h2>课件思维导图</h2>
        <p class="muted">页面展示可点击文字版导图；图片版使用 SVG 下载链接，图太大时会按章节拆分。</p>
      </div>
      <div id="mindmapBody" class="mindmap-grid"></div>
    </section>

    <section id="view-beginner" class="view">
      <div class="panel">
        <div class="quiz-head">
          <div>
            <h2 id="beginnerTitle">基础练习</h2>
            <p id="beginnerMeta" class="muted">适合先熟悉课程脉络。</p>
          </div>
          <button id="beginnerAll" class="btn">重新生成基础题</button>
        </div>
        <div id="beginnerBody"></div>
      </div>
    </section>

    <section id="view-quiz" class="view">
      <div class="panel">
        <div class="quiz-head">
          <div>
            <h2 id="quizTitle">标准综合随机题</h2>
            <p id="quizMeta" class="muted">保留原标准题库，适合考前检测。</p>
          </div>
          <button id="quizAll" class="btn">生成标准题</button>
        </div>
        <div id="quizBody"></div>
      </div>
    </section>

    <section id="view-review" class="view">
      <div class="panel">
        <h2>综合复习资料</h2>
        <p class="muted">这一页保留章节化讲义；从零开始复习建议优先看“教材目录”。</p>
        <div id="reviewBody"></div>
      </div>
    </section>
  </main>
  <script src="assets/course_data.js"></script>
  <script src="assets/course_site.js"></script>
</body>
</html>
"""


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    DATABASE.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    MINDMAP_OUT.mkdir(parents=True, exist_ok=True)

    standard_quiz = load_standard_quiz()
    manifest, chunks, source_cache = extract_sources()
    terms = build_term_index(chunks)
    outline = build_course_outline(manifest, chunks, concept_lookup_from_quiz(standard_quiz))
    review = load_review_or_fallback(outline)
    mindmaps = build_mindmaps(outline)
    beginner_quiz = build_beginner_quiz(outline)

    for img in (ROOT / "image").glob("*.png"):
        if img.name.startswith("截屏"):
            continue
        shutil.copy2(img, IMAGE_OUT / img.name)

    write_json(DATABASE / "source_manifest.json", manifest)
    write_json(DATABASE / "source_cache.json", source_cache)
    write_json(DATABASE / "course_chunks.json", chunks)
    write_json(DATABASE / "course_terms.json", terms)
    write_json(DATABASE / "course_outline.json", outline)
    write_json(DATABASE / "review_material.json", review)
    write_json(DATABASE / "quiz_bank.json", standard_quiz)
    write_json(DATABASE / "beginner_quiz_bank.json", beginner_quiz)
    write_json(DATABASE / "mindmaps.json", mindmaps)

    bundle = {
        "manifest": manifest,
        "chunks": chunks,
        "terms": terms,
        "outline": outline,
        "review": review,
        "mindmaps": mindmaps,
        "quizBank": standard_quiz,
        "beginnerQuizBank": beginner_quiz,
    }
    (ASSETS / "course_data.js").write_text("window.COURSE_DB = " + json.dumps(bundle, ensure_ascii=False) + ";\n", encoding="utf-8")
    (ASSETS / "course_site.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (ASSETS / "course_site.js").write_text(JS.strip() + "\n", encoding="utf-8")
    (SITE / "index.html").write_text(HTML, encoding="utf-8")

    print(
        f"sources={len(manifest)} chunks={len(chunks)} terms={len(terms)} "
        f"standard_quiz={len(standard_quiz)} beginner_quiz={len(beginner_quiz)} mindmaps={sum(len(m['maps']) for m in mindmaps)}"
    )
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
