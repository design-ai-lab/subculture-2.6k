"""
Label check — bilingual (EN/中文) coder tutorial for naive second coders.

Six sections (one per subculture): the paper's palette descriptors plus a
short bilingual description, and N_EXAMPLES corpus images per class sampled
EXCLUDING the 300-image audit sample (verified against answer_key.csv — the
tutorial must not leak audit answers). Examples are copied into
label_audit/tutorial_images/ so the tutorial travels with the zip.

Output: label_audit/tutorial.html (+ tutorial_images/). This file is part of
the disclosed method: "naive coder trained on a standardized visual tutorial".
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under
for _d in (
    CODE_ROOT / "results" / "laion_profile",
    CODE_ROOT / "results" / "separability",
    CODE_ROOT / "results" / "matched_text",
    CODE_ROOT / "dataset",
    CODE_ROOT / "scoring",
):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))


import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
AUDIT = HERE / "label_audit"
IMG_OUT = AUDIT / "tutorial_images"
N_EXAMPLES = 8
SEED = 7

# EN description | ZH description  (palette line from paper section 3, Table 1)
CONTENT = {
    "cottagecore": (
        "Pastoral, romanticized rural life. Pastel and warm earth tones, soft light. "
        "Cottages, wildflower meadows, picnics, baking, vintage dresses, gardens.",
        "田园牧歌式的乡村生活想象。粉彩与温暖的大地色调、柔和光线。"
        "常见元素：乡村小屋、野花草地、野餐、烘焙、复古连衣裙、花园。"),
    "cyberpunk": (
        "Neon-lit night cities, dark and high-contrast, cyan/magenta glow. "
        "Rainy streets, holograms, dense signage, futuristic tech dystopia.",
        "霓虹夜城，画面深暗、高对比，青色/品红光效。"
        "常见元素：雨夜街道、全息投影、密集招牌、未来科技反乌托邦感。"),
    "dark_academia": (
        "Scholarly, moody, nostalgic. Sepia, brown, burgundy; muted, low saturation. "
        "Libraries, old books, tweed and wool, gothic architecture, candlelight.",
        "书卷气、阴郁、怀旧。棕褐、深棕、酒红；整体低饱和、色调压抑。"
        "常见元素：图书馆、旧书、粗花呢与羊毛衣物、哥特式建筑、烛光。"),
    "goblincore": (
        "Earthy forest-floor aesthetic. Brown-green palette, damp and mossy, mystical. "
        "Mushrooms, moss, snails, frogs, gnarled wood, collected trinkets and clutter.",
        "泥土气息的森林地表美学。棕绿色调、潮湿苔藓感、带神秘气质。"
        "常见元素：蘑菇、苔藓、蜗牛、青蛙、虬结的树木、收集的小物件与杂物。"),
    "synthwave": (
        "1980s retro-futurism. Neon magenta/cyan/purple, gradient sunsets, grids. "
        "Grid horizons, chrome lettering, sports cars, palm silhouettes, VHS glow.",
        "80年代复古未来主义。霓虹品红/青/紫、渐变落日、网格。"
        "常见元素：网格地平线、铬金属字体、跑车、棕榈剪影、VHS 光晕。"),
    "y2k": (
        "Turn-of-the-millennium gloss. Chrome, pink, blue; bright, glossy, playful. "
        "Translucent plastic, bubble shapes, early-2000s gadgets and fashion, lens flare.",
        "千禧年之交的光泽感。铬银、粉色、蓝色；明亮、高光泽、俏皮。"
        "常见元素：半透明塑料、泡泡造型、2000年前后的电子产品与时尚、镜头光晕。"),
}


def main() -> None:
    meta = pd.read_csv(REPO_ROOT / "data" / "clip_metadata.csv")
    key = pd.read_csv(AUDIT / "answer_key.csv")
    audit_set = set(zip(key["subculture"], key["filename"]))

    IMG_OUT.mkdir(exist_ok=True)
    sections, n_checked = [], 0
    for sub, (en, zh) in CONTENT.items():
        pool = meta[meta.subculture == sub]
        pool = pool[~pool.apply(lambda r: (r["subculture"], r["filename"]) in audit_set, axis=1)]
        picks = pool.sample(N_EXAMPLES, random_state=SEED)
        # hard verification: no tutorial example may be in the audit sample
        for _, r in picks.iterrows():
            assert (r["subculture"], r["filename"]) not in audit_set, r["filename"]
            n_checked += 1
        imgs = []
        for i, (_, r) in enumerate(picks.iterrows()):
            dst = f"{sub}_{i}.jpg"
            shutil.copyfile(REPO_ROOT / r["relative_path"], IMG_OUT / dst)
            imgs.append(f'<img src="tutorial_images/{dst}" loading="lazy">')
        title = sub.replace("_", " ").title()
        zh_name = {
            "cottagecore": "田园核",
            "cyberpunk": "赛博朋克",
            "dark_academia": "暗黑学院",
            "goblincore": "哥布林核",
            "synthwave": "合成波",
            "y2k": "千禧风 Y2K",
        }[sub]
        sections.append(
            f'<section><h2>{title} <span class="zh">{zh_name}</span></h2>'
            f'<div class="desc"><p>{en}</p><p class="zh">{zh}</p></div>'
            f'<div class="strip">{"".join(imgs)}</div></section>')

    html = """<!doctype html><meta charset='utf-8'><title>Coding tutorial 标注教程</title>
<style>
body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px} h2{margin:26px 0 6px;border-bottom:2px solid #333;padding-bottom:4px}
.zh{color:#555;font-weight:normal}
.desc{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.desc p{margin:4px 0;font-size:14.5px;line-height:1.5}
.strip{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.strip img{width:100%;height:150px;object-fit:cover;border-radius:4px}
.box{background:#f5f5f5;border-left:4px solid #333;padding:10px 14px;margin:14px 0;
     display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:14.5px;line-height:1.5}
</style>
<h1>Image style coding — tutorial <span class="zh">图像风格标注教程</span></h1>
<div class="box">
<div><b>Your task.</b> You will see 300 images. For each, pick the ONE style
(from the six below) that best matches the image, or "unclear" if none fits or
you cannot decide. Judge only what you see in the image — colors, mood,
objects. First impressions are fine (~15 seconds each). There are no trick
questions and "unclear" is always acceptable. Please do not discuss images
with anyone until you finish, and do not look anything up.</div>
<div class="zh"><b>你的任务：</b>你将看到 300 张图片。请为每张图从下面六种风格中
选出最匹配的一种；如果都不符合或无法判断，选 "unclear"（不确定）。只根据画面
本身判断——颜色、氛围、物品。凭第一印象即可（每张约 15 秒）。没有陷阱题，
选"不确定"完全没问题。完成前请不要与任何人讨论图片，也不要上网查询。</div>
</div>
<div class="box">
<div><b>Note.</b> All images (these examples and the 300 you will code) are
AI-generated. Some styles are similar to each other (e.g., the two
forest-and-nature styles, or the neon-city styles) — that is expected; choose
the closest one.</div>
<div class="zh"><b>说明：</b>所有图片（包括这些示例和你要标注的 300 张）都是
AI 生成的。有些风格彼此相近（比如两种森林自然风、或两种霓虹夜景风）——这很
正常，选最接近的即可。</div>
</div>
""" + "".join(sections) + """
<div class="box" style="margin-top:24px">
<div><b>When ready:</b> open <code>coding_sheet.html</code>, type your name in
the coder ID box, and start. You can keep this tutorial open in another tab.</div>
<div class="zh"><b>准备好后：</b>打开 <code>coding_sheet.html</code>，在
coder ID 框中输入你的名字，开始标注。标注时可以在另一个标签页保留本教程。</div>
</div>"""

    (AUDIT / "tutorial.html").write_text(html)
    print(f"[done] tutorial.html — 6 classes x {N_EXAMPLES} examples, "
          f"{n_checked} exclusion checks passed (no overlap with audit sample)")


if __name__ == "__main__":
    main()
