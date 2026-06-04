#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
GUIDES = SITE / "topic_guides"
OUT = SITE / "guide_toc_images"

TOPICS = [
    ("mlp", "MLP 学习资料", "MLP学习资料-通俗版.md"),
    ("cnn", "CNN 学习资料", "CNN学习资料-通俗版.md"),
    ("rnn", "RNN 学习资料", "RNN学习资料-通俗版.md"),
    ("transformer", "Transformer 学习资料", "Transformer学习资料-通俗版.md"),
    ("graph", "图学习资料", "图学习资料-通俗版.md"),
    ("nlp", "NLP 扩展学习资料", "NLP学习资料-通俗版.md"),
    ("llm", "LLM 扩展学习资料", "LLM学习资料-通俗版.md"),
]


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=1 if bold and path.endswith(".ttc") else 0)
    return ImageFont.load_default()


def wrap_text(draw, text, fnt, max_width):
    lines = []
    current = ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial, font=fnt) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def extract_headings(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    headings = []
    for mark, title in re.findall(r"^(#{2,3})\s+(.+)$", text, flags=re.M):
        clean = re.sub(r"\s+", " ", title).strip()
        if clean.startswith(("资料来源", "全局学习路线图")):
            continue
        if "自测题" in clean or "总复习" in clean:
            headings.append((len(mark), clean))
            break
        headings.append((len(mark), clean))
        if len(headings) >= 15:
            break
    return headings[:15]


def locate_base():
    generated_root = Path.home() / ".codex" / "generated_images"
    images = sorted(generated_root.glob("**/*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return images[0] if images else None


def draw_toc(topic_id, title, md_name, base_path):
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = 1400, 1900
    if base_path and base_path.exists():
        img = Image.open(base_path).convert("RGB").resize((w, h))
        img = Image.blend(Image.new("RGB", (w, h), "white"), img, 0.36)
    else:
        img = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(img)
    blue = "#1f8ec7"
    ink = "#1f2937"
    muted = "#667085"
    line = "#d7e4ee"
    draw.rounded_rectangle((70, 70, w - 70, h - 70), radius=28, outline=line, width=3, fill=(255, 255, 255))
    draw.rectangle((70, 70, 112, h - 70), fill="#eaf7fd")
    draw.line((165, 270, w - 150, 270), fill=line, width=3)
    draw.text((170, 118), title, font=font(56, True), fill=ink)
    draw.text((172, 195), "期末复习讲义目录", font=font(28), fill=blue)
    draw.text((1030, 135), "Deep Learning", font=font(24), fill="#94a3b8")

    headings = extract_headings(GUIDES / md_name)
    body_font = font(30)
    small_font = font(23)
    y = 330
    chapter_no = 0
    for level, heading in headings:
        left = 175 if level == 2 else 230
        if level == 2:
            chapter_no += 1
            prefix = f"{chapter_no:02d}"
        else:
            prefix = "·"
        draw.text((left, y), prefix, font=small_font, fill=blue if level == 2 else "#8aa7bb")
        lines = wrap_text(draw, heading, body_font, 870)
        for line in lines[:2]:
            draw.text((left + 88, y - 5), line, font=body_font, fill=ink if level == 2 else muted)
            y += 42
        y += 16
        draw.line((left + 88, y, w - 190, y), fill="#eef2f6", width=2)
        y += 25
        if y > h - 250:
            break

    footer = "先建立直觉，再理解公式、代码、训练流程和考试易错点"
    draw.rounded_rectangle((170, h - 210, w - 170, h - 130), radius=20, fill="#f1f8fc", outline="#cce7f5", width=2)
    draw.text((210, h - 188), footer, font=font(27), fill=blue)
    out = OUT / f"{topic_id}-toc.png"
    img.save(out, quality=95)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = locate_base()
    if base:
        shutil.copy2(base, OUT / "image2-textbook-toc-background.png")
    made = []
    for topic_id, title, md_name in TOPICS:
        made.append(draw_toc(topic_id, title, md_name, base))
    print("\n".join(str(p) for p in made))


if __name__ == "__main__":
    main()
