"""Serve a local HTML viewer for the symbol-filter eval results.

The composite JPGs produced by ``scripts/eval_symbol_filter.py`` cram a lot
of color-coded information into one image. When you zoom in, the legend
strip scrolls off-screen and it becomes hard to remember what each color
means. This script wraps those JPGs in a small browser viewer that:

  - keeps a STICKY LEGEND at the top of the page (color swatches + plain-
    English descriptions, always visible no matter how deep you scroll);
  - shows a SIDEBAR with all images and their per-image stats (dice
    before/after, length ratio change, components dropped, FN added),
    click any image to load it in the main canvas;
  - lets you ZOOM IN/OUT with the mouse wheel and PAN with click-and-drag;
  - shows a TOOLTIP under the cursor when you hover over an overlay color,
    telling you which category the pixel belongs to. The mapping is done
    by sampling the canvas pixel and matching to the known overlay colors
    with a small tolerance (JPG compression slightly shifts pure colors).

The script is self-contained. It reads the existing
``results/v2b_with_symbol_filter/summary.csv`` and the JPGs in that
folder's ``valid/`` (or other split) directories, generates a single
``viewer.html`` next to the CSV, then starts a Python http.server pointing
at that directory and opens the page in your default browser.

Usage
-----
    python -m scripts.start_viewer
    python -m scripts.start_viewer --results-dir results/v2b_with_symbol_filter --port 8765

Press Ctrl-C to stop the server. Page reloads automatically reflect
re-runs of ``scripts.eval_symbol_filter`` (just re-run this script after).
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# The HTML template. Embedded inline so the script is single-file.
# ---------------------------------------------------------------------------

# We use Python string substitution (not f-strings) so we don't have to
# escape every brace in the embedded CSS/JS.
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lateral-detection filter results</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; background: #1b1b1f; color: #eee; }
  body { display: flex; flex-direction: column; height: 100vh; }

  /* Sticky legend bar */
  #legend {
    flex: 0 0 auto; background: #111; border-bottom: 1px solid #444;
    padding: 8px 14px; display: flex; align-items: center; gap: 14px;
    flex-wrap: wrap; font-size: 12.5px;
  }
  #legend .item { display: inline-flex; align-items: center; gap: 6px; cursor: help; }
  #legend .swatch { display: inline-block; width: 18px; height: 14px; border: 1px solid #444; }
  #legend .title  { font-weight: bold; margin-right: 6px; color: #bbb; }
  #legend .note   { color: #888; margin-left: 12px; font-style: italic; }

  /* Main area: sidebar + viewer */
  #main { flex: 1 1 auto; display: flex; min-height: 0; }
  #sidebar {
    flex: 0 0 320px; background: #222; border-right: 1px solid #444;
    overflow-y: auto; font-size: 12.5px;
  }
  #sidebar .header { padding: 10px 12px; font-weight: bold; color: #aaa; border-bottom: 1px solid #333; }
  #sidebar .row {
    padding: 8px 12px; border-bottom: 1px solid #333; cursor: pointer;
    transition: background 80ms ease;
  }
  #sidebar .row:hover  { background: #2c2c30; }
  #sidebar .row.active { background: #3a3a45; }
  #sidebar .row .name  { font-size: 11px; color: #ddd; word-break: break-all; line-height: 1.2; }
  #sidebar .row .stats { margin-top: 6px; color: #aaa; font-size: 11.5px; line-height: 1.4; }
  #sidebar .row .delta.pos { color: #6fdf6f; }
  #sidebar .row .delta.neg { color: #ff7777; }
  #sidebar .row .delta.zero { color: #888; }

  /* Canvas viewer */
  #viewer { flex: 1 1 auto; position: relative; overflow: hidden; background: #0a0a0d; cursor: grab; }
  #viewer.dragging { cursor: grabbing; }
  #canvas { position: absolute; top: 0; left: 0; image-rendering: pixelated; }
  #info { position: absolute; top: 8px; right: 12px; background: rgba(0,0,0,0.7); padding: 6px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; }

  /* Hover tooltip */
  #tooltip {
    position: fixed; pointer-events: none; background: rgba(0,0,0,0.92);
    border: 1px solid #555; border-radius: 4px; padding: 6px 9px; font-size: 12px;
    color: #fff; max-width: 360px; line-height: 1.3; display: none; z-index: 9999;
  }
  #tooltip .swatch {
    display: inline-block; width: 14px; height: 14px; border: 1px solid #888;
    vertical-align: middle; margin-right: 6px;
  }
  #tooltip .cat  { font-weight: bold; }
  #tooltip .desc { color: #ccc; margin-top: 4px; font-size: 11.5px; }
  #tooltip .rgb  { color: #888; font-size: 10.5px; margin-top: 4px; font-family: ui-monospace, Menlo, monospace; }
</style>
</head>
<body>

<div id="legend"><span class="title">Color key (always visible):</span></div>

<div id="main">
  <div id="sidebar"><div class="header">Images (click to view)</div></div>
  <div id="viewer">
    <canvas id="canvas" width="800" height="600"></canvas>
    <div id="info">no image loaded</div>
    <div id="tooltip"></div>
  </div>
</div>

<script>
"use strict";

// ----- categories: color + meaning ------------------------------------------
// Order matters: more specific / smaller-area categories first so they win in
// ambiguous cases (pure red is BOTH "TP dropped" line fill AND the border of
// a non-irrigation symbol rectangle; we list both options when matched).

const CATEGORIES = [
  { name: "TP",            color: [255, 255, 255], desc: "True positive — model and GT agree, filter kept" },
  { name: "FN",            color: [  0, 255, 255], desc: "False negative — GT has a lateral, model didn't predict it. Filter cannot recover these." },
  { name: "FP kept",       color: [255,   0, 255], desc: "False positive that the filter let through. Model predicted, GT says no, filter saw insufficient evidence to drop." },
  { name: "FP dropped",    color: [255, 165,   0], desc: "FILTER WIN — model predicted, GT says no, filter dropped it (e.g., callout fragment with non-irrigation endpoint)." },
  { name: "TP dropped",    color: [255,   0,   0], desc: "FILTER DAMAGE — filter incorrectly removed a real lateral. Should be rare with current settings (min component length protection)." },
  { name: "Symbol: irrigation",   color: [  0, 255,   0], desc: "Symbol with classifier P >= 0.85. An endpoint touching this symbol KEEPS the component." },
  { name: "Symbol: non-irrigation", color: [255,   0,   0], desc: "Symbol with classifier P <= 0.05. Touching only non-irrigation endpoints triggers a drop (subject to length protection)." },
  { name: "Symbol: neutral",        color: [255, 255,   0], desc: "Symbol with classifier P in the middle. No filter action." },
];

// Background / image-content color sampling tolerance.
const COLOR_TOLERANCE = 28;

// Embedded image manifest filled in by start_viewer.py.
const IMAGES = __IMAGES_JSON__;

// ----- build the legend ------------------------------------------------------
const legendEl = document.getElementById("legend");
for (const cat of CATEGORIES) {
  const item = document.createElement("span");
  item.className = "item";
  item.title = cat.desc;
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = `rgb(${cat.color[0]}, ${cat.color[1]}, ${cat.color[2]})`;
  item.appendChild(swatch);
  const label = document.createElement("span");
  label.textContent = cat.name;
  item.appendChild(label);
  legendEl.appendChild(item);
}
const legendNote = document.createElement("span");
legendNote.className = "note";
legendNote.textContent = "(hover any pixel for context. Symbol boxes are rectangle outlines drawn on top of the line categories.)";
legendEl.appendChild(legendNote);

// ----- sidebar ---------------------------------------------------------------
const sidebarEl = document.getElementById("sidebar");
function fmtDelta(v) {
  const s = (v >= 0 ? "+" : "") + v.toFixed(4);
  const cls = v > 0.0005 ? "pos" : (v < -0.0005 ? "neg" : "zero");
  return `<span class="delta ${cls}">${s}</span>`;
}
function fmtNum(v) { return Number(v).toLocaleString(); }

for (let i = 0; i < IMAGES.length; i++) {
  const im = IMAGES[i];
  const row = document.createElement("div");
  row.className = "row"; row.dataset.idx = String(i);
  row.innerHTML = `
    <div class="name">${im.name}</div>
    <div class="stats">
      dice: ${im.dice_b.toFixed(4)} → ${im.dice_a.toFixed(4)} (${fmtDelta(im.dice_d)})<br>
      len_skel ratio: ${im.len_b.toFixed(3)} → ${im.len_a.toFixed(3)}<br>
      dropped: ${im.n_drop} comps, ${fmtNum(im.px_drop)} px;
      FN added: ${fmtNum(im.fn_added)}
    </div>`;
  row.addEventListener("click", () => loadImage(i));
  sidebarEl.appendChild(row);
}

// ----- canvas + interaction --------------------------------------------------
const viewerEl  = document.getElementById("viewer");
const canvas    = document.getElementById("canvas");
const ctx       = canvas.getContext("2d", { willReadFrequently: true });
const tooltip   = document.getElementById("tooltip");
const infoEl    = document.getElementById("info");

// World transform: image pixels mapped to viewer pixels by (offsetX + sx * scale, ...).
let currentImage = null;
let imgEl        = null;
let scale        = 1.0;
let offX         = 0;
let offY         = 0;
let isDragging   = false;
let dragStartX   = 0;
let dragStartY   = 0;
let origOffX     = 0;
let origOffY     = 0;

function fitImageToViewer() {
  if (!imgEl) return;
  const vw = viewerEl.clientWidth;
  const vh = viewerEl.clientHeight;
  const sx = vw / imgEl.naturalWidth;
  const sy = vh / imgEl.naturalHeight;
  scale = Math.min(sx, sy) * 0.98;
  offX  = (vw - imgEl.naturalWidth  * scale) / 2;
  offY  = (vh - imgEl.naturalHeight * scale) / 2;
  redraw();
}

function redraw() {
  if (!imgEl) return;
  const vw = viewerEl.clientWidth;
  const vh = viewerEl.clientHeight;
  canvas.width  = vw;
  canvas.height = vh;
  ctx.fillStyle = "#0a0a0d";
  ctx.fillRect(0, 0, vw, vh);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(imgEl, offX, offY, imgEl.naturalWidth * scale, imgEl.naturalHeight * scale);
  infoEl.textContent = `${currentImage.name}  —  ${imgEl.naturalWidth}×${imgEl.naturalHeight}px  zoom ${(scale * 100).toFixed(0)}%`;
}

window.addEventListener("resize", redraw);

function loadImage(idx) {
  // Highlight the active sidebar row
  for (const row of sidebarEl.querySelectorAll(".row")) {
    row.classList.toggle("active", Number(row.dataset.idx) === idx);
  }
  currentImage = IMAGES[idx];
  imgEl = new Image();
  imgEl.onload = () => { fitImageToViewer(); };
  imgEl.src = currentImage.jpg;
  tooltip.style.display = "none";
}

// Auto-load first image
if (IMAGES.length > 0) loadImage(0);

// Zoom (wheel)
viewerEl.addEventListener("wheel", (e) => {
  if (!imgEl) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  // Image pixel under cursor BEFORE zoom
  const ix = (mx - offX) / scale;
  const iy = (my - offY) / scale;
  const factor = (e.deltaY < 0) ? 1.15 : 1 / 1.15;
  scale = Math.max(0.05, Math.min(20.0, scale * factor));
  // Keep the same image pixel under the cursor
  offX = mx - ix * scale;
  offY = my - iy * scale;
  redraw();
}, { passive: false });

// Drag-pan
viewerEl.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  isDragging = true; viewerEl.classList.add("dragging");
  dragStartX = e.clientX; dragStartY = e.clientY;
  origOffX = offX; origOffY = offY;
  tooltip.style.display = "none";
});
window.addEventListener("mouseup", () => {
  isDragging = false; viewerEl.classList.remove("dragging");
});
window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  offX = origOffX + (e.clientX - dragStartX);
  offY = origOffY + (e.clientY - dragStartY);
  redraw();
});

// Hover tooltip (pixel sampling)
function matchCategories(rgb) {
  const matches = [];
  for (const cat of CATEGORIES) {
    const [cr, cg, cb] = cat.color;
    const dr = Math.abs(rgb[0] - cr);
    const dg = Math.abs(rgb[1] - cg);
    const db = Math.abs(rgb[2] - cb);
    if (dr <= COLOR_TOLERANCE && dg <= COLOR_TOLERANCE && db <= COLOR_TOLERANCE) {
      matches.push(cat);
    }
  }
  return matches;
}

viewerEl.addEventListener("mousemove", (e) => {
  if (isDragging || !imgEl) { tooltip.style.display = "none"; return; }
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  if (mx < 0 || my < 0 || mx >= canvas.width || my >= canvas.height) {
    tooltip.style.display = "none"; return;
  }
  let pixel;
  try { pixel = ctx.getImageData(mx, my, 1, 1).data; } catch (err) { return; }
  const matches = matchCategories(pixel);
  if (matches.length === 0) {
    tooltip.style.display = "none";
    return;
  }
  let html = "";
  for (const cat of matches) {
    html += `<div>
      <span class="swatch" style="background: rgb(${cat.color[0]}, ${cat.color[1]}, ${cat.color[2]})"></span>
      <span class="cat">${cat.name}</span>
      <div class="desc">${cat.desc}</div>
    </div>`;
  }
  html += `<div class="rgb">rgb(${pixel[0]}, ${pixel[1]}, ${pixel[2]})</div>`;
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  // Place tooltip near cursor, but flip if it would go off-screen.
  const tw = tooltip.offsetWidth; const th = tooltip.offsetHeight;
  let tx = e.clientX + 14; let ty = e.clientY + 14;
  if (tx + tw > window.innerWidth - 8)  tx = e.clientX - 14 - tw;
  if (ty + th > window.innerHeight - 8) ty = e.clientY - 14 - th;
  tooltip.style.left = tx + "px";
  tooltip.style.top  = ty + "px";
});

viewerEl.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Manifest building (CSV → JSON for the embedded JS)
# ---------------------------------------------------------------------------


def _gather_images(results_dir: Path) -> list[dict]:
    csv_path = results_dir / "summary.csv"
    if not csv_path.is_file():
        print(f"[viewer] no summary.csv at {csv_path}", file=sys.stderr)
        return []
    images: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            split = r["split"]
            stem  = Path(r["image_file"]).stem
            jpg   = results_dir / split / f"{stem}.jpg"
            if not jpg.is_file():
                continue
            images.append({
                "name":     r["image_file"],
                "jpg":      jpg.relative_to(results_dir).as_posix(),
                "dice_b":   float(r["dice_before"]),
                "dice_a":   float(r["dice_after"]),
                "dice_d":   float(r["dice_delta"]),
                "len_b":    float(r["len_skel_ratio_before"]),
                "len_a":    float(r["len_skel_ratio_after"]),
                "n_drop":   int(r["n_dropped"]),
                "px_drop":  int(r["pixels_dropped"]),
                "fn_added": int(r["fn_added"]),
            })
    return images


def write_viewer_html(results_dir: Path) -> Path:
    images = _gather_images(results_dir)
    html = HTML_TEMPLATE.replace("__IMAGES_JSON__", json.dumps(images))
    out_path = results_dir / "viewer.html"
    out_path.write_text(html)
    print(f"[viewer] wrote {out_path}  ({len(images)} images indexed)")
    return out_path


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    # Suppress per-request log lines (keeps the terminal clean).
    def log_message(self, format, *args):  # noqa: A003
        pass


def serve(results_dir: Path, port: int, open_browser: bool = True) -> None:
    os.chdir(results_dir)
    handler = _QuietHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/viewer.html"
        print(f"[viewer] serving {results_dir} on {url}")
        print(f"[viewer] press Ctrl-C to stop")
        if open_browser:
            # Open the browser a beat after the server is ready.
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[viewer] stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results/v2b_with_symbol_filter",
                        help="Directory containing summary.csv and split subdirs with JPGs.")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind on localhost.")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the browser.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"[viewer] not a directory: {results_dir}", file=sys.stderr)
        return 1

    write_viewer_html(results_dir)
    serve(results_dir, args.port, open_browser=(not args.no_open))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
