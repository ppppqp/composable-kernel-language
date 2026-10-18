#!/usr/bin/env python3
"""Validate Nsight SQLite exports; write portable traces and GPU timeline figures.

Uses only the Python standard library. Timestamps come from CUPTI activity,
never from the compiler's predicted schedule. Packing lanes are visualization
rows, not CUDA streams or SMs; original stream/graph IDs remain in trace args.
"""
import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import heapq
from html import escape
import json
from pathlib import Path
import sqlite3


COLORS = dict(linear="#2563eb", rms_stats="#9333ea", rms_apply="#c084fc",
              rope="#0d9488", append_kv="#14b8a6", attention_scores="#ea580c",
              softmax="#f59e0b", attention_values="#f97316", greedy="#dc2626")


def load(prefix):
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if not metadata["complete"]:
        raise ValueError(f"incomplete profile: {prefix}")
    connection = sqlite3.connect(f"file:{prefix.with_suffix('.sqlite')}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    strings = dict(connection.execute("select id,value from StringIds"))
    kernels = [dict(row) for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start,end")]
    apis = [dict(row) for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    count = metadata["plan"]["nodes"] * metadata["config"]["steps"] * metadata["requests"]
    if len(kernels) != count:
        raise ValueError(f"{prefix}: expected {count} GPU kernels, found {len(kernels)}")
    if any(row["end"] <= row["start"] for row in kernels) or any(row["returnValue"] for row in apis):
        raise ValueError(f"{prefix}: invalid timestamps or failed CUDA API")
    copies = sum(connection.execute(f"select count(*) from {table}").fetchone()[0]
                 for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET") if table in tables)
    if copies:
        raise ValueError("input preparation leaked into profile range")
    api_counts = Counter(strings[row["nameId"]] for row in apis)
    expected_launches = metadata["requests"] * metadata["config"]["steps"] // metadata["group"]
    if metadata["mode"] == "ordinary":
        assert api_counts["cuLaunchKernel"] == count and not api_counts["cuGraphLaunch"]
    else:
        assert api_counts["cuGraphLaunch"] == expected_launches and not api_counts["cuLaunchKernel"]
    origin = min(row["start"] for row in kernels)
    stop = max(row["end"] for row in kernels)
    changes = defaultdict(int)
    families = defaultdict(float)
    lane_ends = []
    trace = []
    for row in kernels:
        symbol = strings[row["demangledName"]]
        if symbol not in metadata["symbols"]:
            raise ValueError(f"unknown kernel {symbol}")
        family = metadata["symbols"][symbol]
        # Reuse an available lane. These lanes expose actual interval overlap.
        if lane_ends and lane_ends[0][0] <= row["start"]:
            _, lane = heapq.heappop(lane_ends)
        else:
            lane = len(lane_ends)
        heapq.heappush(lane_ends, (row["end"], lane))
        begin, end = (row["start"] - origin) / 1000, (row["end"] - origin) / 1000
        changes[row["start"]] += 1
        changes[row["end"]] -= 1
        families[family] += end - begin
        args = {key: row[key] for key in ("streamId", "correlationId", "graphNodeId", "graphId",
                                         "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                                         "registersPerThread") if key in row}
        args["symbol"] = symbol
        trace.append(dict(name=family, cat="GPU kernel", ph="X", pid=1, tid=lane,
                          ts=begin, dur=end - begin, args=args))
    active, previous, busy, overlap, maximum = 0, origin, 0, 0, 0
    for timestamp, delta in sorted(changes.items()):
        duration = timestamp - previous
        busy += duration if active else 0
        overlap += duration if active > 1 else 0
        active += delta
        maximum = max(maximum, active)
        previous = timestamp
    for row in apis:
        trace.append(dict(name=strings[row["nameId"]], cat="CUDA API", ph="X", pid=2,
                          tid=row["globalTid"], ts=(row["start"] - origin) / 1000,
                          dur=(row["end"] - row["start"]) / 1000,
                          args=dict(correlationId=row["correlationId"])))
    for lane in range(len(lane_ends)):
        trace.append(dict(ph="M", name="thread_name", pid=1, tid=lane,
                          args=dict(name=f"GPU interval lane {lane}")))
    trace += [dict(ph="M", name="process_name", pid=1, args=dict(name="GPU (packed intervals)")),
              dict(ph="M", name="process_name", pid=2, args=dict(name="Host CUDA APIs"))]
    for tid in {row["globalTid"] for row in apis}:
        trace.append(dict(ph="M", name="thread_name", pid=2, tid=tid,
                          args=dict(name=f"CPU thread {tid & 0xffffff}")))
    # Shift negative host timestamps to zero for portable Chrome trace viewers.
    offset = -min(event.get("ts", 0) for event in trace)
    chrome = [dict(event, ts=event["ts"] + offset) if "ts" in event else event for event in trace]
    with gzip.GzipFile(filename=str(prefix.with_suffix(".trace.json.gz")), mode="wb", mtime=0) as handle:
        handle.write(json.dumps(dict(traceEvents=chrome, displayTimeUnit="us"), separators=(",", ":")).encode())
    span = (stop - origin) / 1000
    summary = dict(name=prefix.name, kernels=count, span_us=span, busy_us=busy / 1000,
                   idle_us=(stop - origin - busy) / 1000, overlap_us=overlap / 1000,
                   overlap_fraction=overlap / (stop - origin), max_concurrent_kernels=maximum,
                   sum_kernel_us=sum(families.values()), family_sum_kernel_us=dict(sorted(families.items(), key=lambda x: -x[1])),
                   cuda_api_counts=dict(api_counts), copies_or_memsets=copies,
                   sqlite_sha256=hashlib.sha256(prefix.with_suffix(".sqlite").read_bytes()).hexdigest(),
                   first_step_end_us=max(e["ts"] + e["dur"] for e in trace[:metadata["plan"]["nodes"]]),
                   max_abs_error=metadata["max_abs_error"])
    connection.close()
    return dict(summary=summary, metadata=metadata, events=[e for e in trace if e.get("ph") == "X"])


def svg(path, profiles, names):
    selected = [p for p in profiles if p["summary"]["name"] in names]
    end = max(p["summary"]["first_step_end_us"] for p in selected) * 1.04
    width, left, plot = 1200, 190, 960
    rows = []
    y = 55
    for profile in selected:
        s = profile["summary"]
        rows.append(f'<text x="10" y="{y+12}" font-size="13">{escape(s["name"])}</text>')
        for event in profile["events"]:
            if event["cat"] != "GPU kernel" or event["ts"] > end:
                continue
            x = left + event["ts"] / end * plot
            w = min(event["dur"], end - event["ts"]) / end * plot
            lane_y = y + event["tid"] * 14
            title = escape(f"{event['name']}: {event['ts']:.3f}–{event['ts']+event['dur']:.3f} µs; CUDA stream {event['args']['streamId']}")
            rows.append(f'<rect x="{x:.2f}" y="{lane_y}" width="{max(w,.4):.2f}" height="11" fill="{COLORS.get(event["name"], "#64748b")}"><title>{title}</title></rect>')
        y += max(2, s["max_concurrent_kernels"]) * 14 + 35
    for i in range(9):
        x = left + plot * i / 8
        rows += [f'<line x1="{x}" x2="{x}" y1="40" y2="{y}" stroke="#cbd5e1" opacity=".45"/>',
                 f'<text x="{x}" y="{y+17}" text-anchor="middle" font-size="11">{end*i/8:.1f}</text>']
    y += 45
    for i, (name, color) in enumerate(COLORS.items()):
        x, yy = 15 + i % 5 * 235, y + i // 5 * 20
        rows += [f'<rect x="{x}" y="{yy}" width="10" height="10" fill="{color}"/>',
                 f'<text x="{x+15}" y="{yy+10}" font-size="11">{name}</text>']
    rows.append('<text x="15" y="23" font-size="16">Actual CUPTI GPU intervals • opening window • horizontal axis: µs from first kernel</text>')
    rows.append(f'<text x="15" y="{y+60}" font-size="12">Rows pack overlapping intervals; they do not represent SMs. Hover rectangles for kernel times and stream IDs.</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y+80}" viewBox="0 0 {width} {y+80}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#0f172a">'+"".join(rows)+"</g></svg>\n")


def html(path, profiles):
    # Embed all data so file:// works without a server or network requests.
    payload = base64.b64encode(gzip.compress(json.dumps(profiles, separators=(",", ":")).encode(), mtime=0)).decode()
    path.write_text('''<!doctype html><meta charset="utf-8"><title>CKL decoder GPU traces</title>
<style>body{font:15px system-ui;margin:28px;color:#172033}label{margin-right:15px}canvas{border:1px solid #ccc;width:100%;height:430px}button,input,select{font:inherit}#details{white-space:pre-wrap;background:#f1f5f9;padding:14px}p{max-width:1000px}</style>
<h1>CKL decoder GPU traces</h1><p>Recorded CUPTI kernel intervals and host CUDA calls. GPU lanes pack simultaneous intervals; they are not SMs. Trace timings include profiler overhead. Use the unprofiled benchmark for speedups. Hover a rectangle for its original stream, graph, launch shape, and timestamps.</p>
<label>Trace <select id="trace"></select></label><button id="full">Full range</button> <button id="step">First step</button><br><br>
<label>Start µs <input id="start" type="number" value="0" min="0" step="10"></label><label>Window µs <input id="window" type="number" min="1" step="10"></label><label>Kernel filter <input id="filter" placeholder="e.g. linear"></label>
<p id="stats"></p><canvas id="canvas"></canvas><pre id="details">Hover a kernel or CUDA call.</pre>
<script type="module">const packed="PAYLOAD";const stream=new Blob([Uint8Array.from(atob(packed),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'));const profiles=JSON.parse(await new Response(stream).text()),colors=COLORS;const $=id=>document.getElementById(id),sel=$('trace'),canvas=$('canvas'),ctx=canvas.getContext('2d');let boxes=[];
profiles.forEach((p,i)=>{let o=document.createElement('option');o.value=i;o.textContent=p.summary.name;sel.appendChild(o)});
function draw(){const p=profiles[+sel.value],s=p.summary;let from=+$('start').value,window=Math.max(1,+$('window').value),filter=$('filter').value.toLowerCase();canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=430*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);let w=canvas.clientWidth;ctx.clearRect(0,0,w,430);ctx.font='12px system-ui';boxes=[];let tids=[...new Set(p.events.filter(e=>e.cat==='CUDA API').map(e=>e.tid))];let laneHeight=Math.min(23,330/(s.max_concurrent_kernels+tids.length));let left=155,pw=w-left-15;
for(let i=0;i<=8;i++){let x=left+pw*i/8;ctx.strokeStyle='#e2e8f0';ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,410);ctx.stroke();ctx.fillStyle='#334155';ctx.fillText((from+window*i/8).toFixed(1),x-12,15)}
for(let i=0;i<tids.length;i++){ctx.fillStyle='#334155';ctx.fillText('CPU '+(tids[i]&0xffffff),4,42+i*laneHeight)}for(let i=0;i<s.max_concurrent_kernels;i++){ctx.fillStyle='#334155';ctx.fillText('GPU interval lane '+i,4,42+(i+tids.length+1)*laneHeight)}
for(const e of p.events){if(e.ts+e.dur<from||e.ts>from+window)continue;if(filter&&e.cat==='GPU kernel'&&!e.name.includes(filter))continue;let lane=e.cat==='CUDA API'?tids.indexOf(e.tid):tids.length+1+e.tid;let x=left+Math.max(0,e.ts-from)/window*pw,y=30+lane*laneHeight,bw=Math.max(.6,(Math.min(from+window,e.ts+e.dur)-Math.max(from,e.ts))/window*pw);ctx.fillStyle=e.cat==='CUDA API'?'#94a3b8':colors[e.name]||'#64748b';ctx.fillRect(x,y,bw,laneHeight-3);boxes.push({x,y,w:bw,h:laneHeight-3,e})}
$('stats').textContent=`${s.kernels} kernels · GPU span ${s.span_us.toFixed(1)} µs · ≥2 kernels active ${(100*s.overlap_fraction).toFixed(1)}% of span · max simultaneous ${s.max_concurrent_kernels} · GPU gaps ${s.idle_us.toFixed(1)} µs`}
function select(){let s=profiles[+sel.value].summary;$('start').value=0;$('window').value=s.first_step_end_us*1.05;draw()}sel.onchange=select;$('full').onclick=()=>{$('start').value=0;$('window').value=profiles[+sel.value].summary.span_us;draw()};$('step').onclick=select;['start','window','filter'].forEach(id=>$(id).oninput=draw);addEventListener('resize',draw);canvas.onmousemove=event=>{let r=canvas.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;let b=boxes.findLast(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);if(b)$('details').textContent=JSON.stringify(b.e,null,2)};select();</script>'''.replace("PAYLOAD", payload).replace("COLORS", json.dumps(COLORS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.traces / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("incomplete collection")
    profiles = [load(args.traces / item["name"]) for item in manifest["profiles"]]
    (args.traces / "summary.json").write_text(json.dumps([p["summary"] for p in profiles], indent=2) + "\n")
    html(args.traces / "viewer.html", profiles)
    for name, names in dict(small=["small_sequential", "small_capture", "small_inferred", "small_reuse", "small_group16"],
                             wide=["wide_capture", "wide_inferred"],
                             long=["long_capture", "long_inferred"],
                             requests=["requests_serial", "requests_parallel", "requests_group16"],
                             vocab=["vocab_inferred"]).items():
        if any(p["summary"]["name"] in names for p in profiles):
            svg(args.traces / f"{name}.svg", profiles, names)
    print(json.dumps([p["summary"] for p in profiles], indent=2))


if __name__ == "__main__":
    main()
