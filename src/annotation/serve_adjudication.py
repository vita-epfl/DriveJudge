#!/usr/bin/env python3
"""Tiny local web UI to fill an adjudication CSV (final label + reason + note).

Usage:
    python src/annotation/serve_adjudication.py results/annotation/adjudication_reliability.csv --port 8765
Then open http://127.0.0.1:8765 . Every click is written straight back into the CSV
(atomic rewrite), so the server can be killed at any time. Nothing else is modified.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

REASONS = ["label_error", "annotator_slip", "ambiguous_clip", "convention"]
REASON_HELP = {
    "label_error": "original GT wrong, annotator 2 right",
    "annotator_slip": "annotator 2 misclicked / missed it, GT stands",
    "ambiguous_clip": "genuinely undecidable, best guess in final",
    "convention": "different reading of the question (anchors, side, wording)",
}
LOCK = threading.Lock()


def load(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def save(path: Path, rows, fieldnames):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".adj_", suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def options_for(row):
    labs = {row["original_label"].strip(), row["annotator2"].strip()}
    if labs <= {"1", "2", "3"}:
        return ["1", "2", "3"]
    if labs <= {"Yes", "No"}:
        return ["Yes", "No"]
    if labs <= {"Left", "Right"}:
        return ["Left", "Right"]
    return sorted(labs)


def is_realism(row):
    return row["question"].lower().startswith("overall, how realistic")


def render(rows, reason_col):
    # order: non-realism first, realism block last (bulk-fillable)
    order = sorted(range(len(rows)), key=lambda i: (is_realism(rows[i]), rows[i]["question"], rows[i]["clip"]))
    n_todo = sum(not is_realism(r) for r in rows)
    n_real = len(rows) - n_todo
    cards = []
    n_r = 0
    for n, i in enumerate(order, 1):
        r = rows[i]
        opts = options_for(r)
        fin = r["final"].strip()
        rsn = r[reason_col].strip()
        realism = is_realism(r)
        final_html = "".join(
            f'<label><input type="radio" name="final_{i}" value="{html.escape(o)}" {"checked" if fin == o else ""} '
            f'onchange="upd({i},\'final\',this.value)"> {html.escape(o)}</label>'
            for o in opts
        )
        reason_html = "".join(
            f'<label title="{html.escape(REASON_HELP[o])}"><input type="radio" name="reason_{i}" value="{o}" {"checked" if rsn == o else ""} '
            f'onchange="upd({i},\'reason\',this.value)"> {o}</label>'
            for o in REASONS
        )
        if realism:
            n_r += 1
            num = f"realism {n_r} / {n_real}"
        else:
            num = f"{n} / {n_todo}"
        cards.append(
            f'<section class="card {"realism" if realism else ""}" id="c{i}" data-i="{i}">'
            f'<div class="head"><span class="num">{num}</span>'
            f'<span class="clip">{html.escape(r["clip"])}</span></div>'
            f'<video src="/video/{html.escape(r["clip"])}" controls loop muted preload="none"></video>'
            f'<div class="q"><b>Q:</b> {html.escape(r["question"])}</div>'
            f'<div class="labels"><span class="gt">original GT: <b>{html.escape(r["original_label"])}</b></span>'
            f'<span class="a2">annotator 2: <b>{html.escape(r["annotator2"])}</b></span></div>'
            f'<div class="row"><span class="lbl">final</span>{final_html}</div>'
            f'<div class="row"><span class="lbl">reason</span>{reason_html}</div>'
            f'<div class="row"><span class="lbl">note</span><input type="text" class="note" value="{html.escape(r["note"])}" '
            f'onchange="upd({i},\'note\',this.value)" placeholder="optional"></div>'
            f"</section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Adjudication</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;background:#fafafa;font-size:16px}}
.bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;z-index:2;display:flex;gap:16px;align-items:center}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px 18px;margin:16px 0}}
.card.done{{border-color:#7c7}} .card.realism{{background:#fdfdf5;display:none}} body.showreal .card.realism{{display:block}}
h2{{margin-top:32px;border-top:2px solid #ccc;padding-top:12px}}
.head{{display:flex;gap:16px;align-items:baseline;font-weight:600}} .num{{font-size:20px}} .clip{{font-weight:400;color:#666;font-size:13px}}
video{{width:100%;max-height:420px;background:#000;border-radius:6px;margin:8px 0}}
.q{{margin:6px 0}} .labels span{{margin-right:24px}} .gt{{color:#146}} .a2{{color:#a40}}
.row{{margin:8px 0;display:flex;align-items:center;flex-wrap:wrap;gap:6px}} .lbl{{width:60px;color:#555}}
label{{padding:4px 10px;border:1px solid #ccc;border-radius:6px;cursor:pointer;background:#fff}} label:hover{{background:#f0f0f0}}
.note{{flex:1;font-size:15px;padding:4px 8px}} button{{font-size:15px;padding:6px 12px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer}}
#status{{color:#555}} .help{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:12px 0;font-size:14px;color:#444}}
</style></head><body>
<div class="bar"><span id="status"></span>
<button onclick="bulkRealism()">Realism block: set final = original GT, reason = convention ({n_real} rows)</button>
<button onclick="nextTodo()">Jump to next unfilled</button>
<button onclick="document.body.classList.toggle('showreal')">Show / hide realism block</button></div>
<h1>Adjudication: {n_todo} clips to watch</h1>
<div class="help">Each card: the clip (click to play), the question, the two labels. Set <b>final</b> (label you now stand behind) and <b>reason</b>:
{"; ".join(f"<b>{k}</b> = {v}" for k, v in REASON_HELP.items())}. Saved to the CSV on every click.
The {n_real} realism rows are hidden (agreed anchor convention): fill them with the bulk button, no need to watch them.</div>
{"".join(cards[:n_todo])}
<h2>Realism block ({n_real} rows, hidden by default)</h2>
{"".join(cards[n_todo:])}
<script>
function status(){{const cards=[...document.querySelectorAll('.card:not(.realism)')];let done=0;cards.forEach(c=>{{const i=c.dataset.i;const f=c.querySelector('input[name=final_'+i+']:checked');const r=c.querySelector('input[name=reason_'+i+']:checked');const ok=f&&r;c.classList.toggle('done',!!ok);if(ok)done++;}});const rc=[...document.querySelectorAll('.card.realism')].filter(c=>{{const i=c.dataset.i;return c.querySelector('input[name=final_'+i+']:checked')&&c.querySelector('input[name=reason_'+i+']:checked');}}).length;document.getElementById('status').textContent='to watch: '+done+' / '+cards.length+' done  |  realism: '+rc+' / '+document.querySelectorAll('.card.realism').length+' filled';}}
async function upd(i,field,value){{const res=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{i:i,field:field,value:value}})}});if(!res.ok){{alert('save failed: '+res.status);}}status();}}
async function bulkRealism(){{const res=await fetch('/bulk_realism',{{method:'POST'}});if(res.ok){{location.reload();}}else alert('failed');}}
function nextTodo(){{const c=[...document.querySelectorAll('.card')].find(c=>!c.classList.contains('done'));if(c)c.scrollIntoView({{behavior:'smooth',block:'start'}});}}
document.querySelectorAll('video').forEach(v=>{{const io=new IntersectionObserver(e=>e.forEach(x=>{{if(x.isIntersecting){{v.preload='auto';}}else{{v.pause();}}}}));io.observe(v);}});
status();
</script></body></html>"""
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--video-root", default=str(Path.home() / "Documents/Projet de master/final_dataset"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    csv_path = Path(args.csv).resolve()
    video_root = Path(args.video_root).resolve()
    rows = load(csv_path)
    fieldnames = list(rows[0].keys())
    reason_col = next(c for c in fieldnames if c.startswith("reason"))

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                with LOCK:
                    body = render(rows, reason_col).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/video/"):
                rel = unquote(self.path[len("/video/"):])
                p = (video_root / rel).resolve()
                if not str(p).startswith(str(video_root)) or not p.is_file():
                    self.send_error(404)
                    return
                self._send_file(p)
                return
            self.send_error(404)

        def _send_file(self, p: Path):
            size = p.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                    if not m.group(1) and m.group(2):
                        start = max(size - int(m.group(2)), 0)
                        end = size - 1
            length = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            with LOCK:
                if self.path == "/save":
                    d = json.loads(raw)
                    i = int(d["i"])
                    field = {"final": "final", "reason": reason_col, "note": "note"}[d["field"]]
                    rows[i][field] = str(d["value"])
                    save(csv_path, rows, fieldnames)
                elif self.path == "/bulk_realism":
                    for r in rows:
                        if is_realism(r):
                            if not r["final"].strip():
                                r["final"] = r["original_label"]
                            if not r[reason_col].strip():
                                r[reason_col] = "convention"
                    save(csv_path, rows, fieldnames)
                else:
                    self.send_error(404)
                    return
            self.send_response(204)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"serving {csv_path.name} ({len(rows)} rows) at http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
