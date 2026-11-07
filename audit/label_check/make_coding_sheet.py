"""
Label check — interactive blinded label-coding sheet (click-to-label, like the
photograph review sheet). Replaces the CSV-by-hand workflow.

Generates label_audit/coding_sheet.html over the existing anonymized sample
(label_audit/sample_images/sample_*.jpg — no labels visible anywhere):

  - enter a coder ID once (top bar); decisions are stored per-coder in
    localStorage, so a second coder on the same machine just types a
    different ID (do NOT discuss items between coders)
  - each image has 7 chips: the six subcultures + "unclear"; click one
  - progress counter; auto-save; "Download" saves labels_coder_<coder>.csv
  - keyboard: hover an image and press 1-6 or u

Afterwards run analyze_label_audit.py (accuracy vs platform labels; Cohen's
kappa when two coders' files exist).
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


from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "label_audit"

LABELS = ["cottagecore", "cyberpunk", "dark_academia",
          "goblincore", "synthwave", "y2k", "unclear"]
COLORS = ["#7a9d54", "#c026d3", "#7c5c2e", "#4d7c0f", "#7c3aed", "#0891b2", "#9ca3af"]

samples = sorted(p.name for p in (AUDIT / "sample_images").glob("sample_*.jpg"))

chip_defs = "".join(
    f'<span class="chip" data-l="{l}" style="--c:{c}">{i + 1 if l != "unclear" else "u"}·{l[:4]}</span>'
    for i, (l, c) in enumerate(zip(LABELS, COLORS)))

cards = "".join(
    f'<figure class="card" data-id="{Path(s).stem}">'
    f'<img src="sample_images/{s}" loading="lazy">'
    f'<div class="chips">{chip_defs}</div>'
    f'<figcaption>{Path(s).stem}</figcaption></figure>'
    for s in samples)

html = """<!doctype html><meta charset='utf-8'><title>Blinded label coding</title>
<style>
body{font-family:sans-serif;margin:0;padding:10px}
#bar{position:sticky;top:0;background:#fff;padding:10px;border-bottom:2px solid #333;
     display:flex;gap:14px;align-items:center;z-index:9;flex-wrap:wrap}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;padding-top:10px}
.card{margin:0;border:3px solid #ddd;border-radius:6px;overflow:hidden}
.card.done{border-color:var(--done)}
.card img{width:100%;height:210px;object-fit:cover;display:block}
.chips{display:flex;flex-wrap:wrap;gap:3px;padding:4px}
.chip{font-size:10.5px;padding:2px 5px;border:1px solid var(--c);color:var(--c);
      border-radius:9px;cursor:pointer;user-select:none}
.chip.sel{background:var(--c);color:#fff}
figcaption{font-size:11px;text-align:center;padding-bottom:4px}
button{font-size:15px;padding:8px 14px;cursor:pointer}
input{font-size:15px;padding:6px}
</style>
<div id="bar">
  <b>Blinded label coding</b>
  <label>coder ID: <input id="coder" placeholder="e.g. coder" size="8"></label>
  <span id="prog"></span>
  <button onclick="download()">⬇ Download</button>
  <span style="color:#666;font-size:13px">click a chip per image (or hover image + keys 1–6, u) ·
  judge the IMAGE, not the query · auto-saves per coder · do not open answer_key.csv</span>
</div>
<div id="grid">""" + cards + """</div>
<script>
const LABELS=""" + str(LABELS) + """;
let coder=localStorage.getItem("label_last_coder")||"";
const coderEl=document.getElementById("coder");
coderEl.value=coder;
let d={};
function key(){return "label_"+coder;}
function load(){d=coder?JSON.parse(localStorage.getItem(key())||"{}"):{};}
function save(){if(coder){localStorage.setItem(key(),JSON.stringify(d));
                localStorage.setItem("label_last_coder",coder);}}
coderEl.onchange=()=>{coder=coderEl.value.trim();load();refresh();};
const cards=[...document.querySelectorAll(".card")];
function refresh(){
  let n=0;
  cards.forEach(c=>{
    const v=d[c.dataset.id];
    c.style.setProperty("--done", v ? "#16a34a" : "#ddd");
    c.classList.toggle("done", !!v);
    c.querySelectorAll(".chip").forEach(ch=>ch.classList.toggle("sel", ch.dataset.l===v));
    if(v) n++;
  });
  document.getElementById("prog").textContent=`labeled ${n}/${cards.length}`;
}
cards.forEach(c=>{
  c.querySelectorAll(".chip").forEach(ch=>ch.onclick=()=>{
    if(!coder){alert("enter a coder ID first");return;}
    d[c.dataset.id]=(d[c.dataset.id]===ch.dataset.l)?undefined:ch.dataset.l;
    if(d[c.dataset.id]===undefined) delete d[c.dataset.id];
    save();refresh();
  });
  c.onmouseenter=()=>{hover=c;};
});
let hover=null;
document.onkeydown=e=>{
  if(!hover||!coder||e.target.tagName==="INPUT")return;
  let l=null;
  if(e.key>="1"&&e.key<="6") l=LABELS[+e.key-1];
  if(e.key==="u") l="unclear";
  if(l){d[hover.dataset.id]=l;save();refresh();}
};
function download(){
  if(!coder){alert("enter a coder ID first");return;}
  const lines=["sample_id,label"];
  cards.forEach(c=>{if(d[c.dataset.id])lines.push(`${c.dataset.id},${d[c.dataset.id]}`);});
  const b=new Blob([lines.join("\\n")],{type:"text/csv"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(b);a.download=`labels_coder_${coder}.csv`;a.click();
}
load();refresh();
</script>"""

(AUDIT / "coding_sheet.html").write_text(html)
print(f"[done] coding_sheet.html — {len(samples)} images, interactive, multi-coder")
