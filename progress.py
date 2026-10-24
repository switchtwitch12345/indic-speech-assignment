#!/usr/bin/env python3
"""Live progress for the whole project. Usage:  python3 -u progress.py      (Ctrl-C to quit)
Reads result files and the job log only; uses almost no CPU/memory. Overall % is weighted by estimated minutes per stage."""
import os, re, subprocess, time

R, LOG = "results", "results/qa_run.log"
# (group, label, file, expected rows, est. minutes, incremental?)  incremental=False -> file is written when the stage ends
S = [
 ("Baseline + first variants (complete)", "whisper-small -> Qwen, clean", "base_clean.jsonl", 120, 8, True),
 ("Baseline + first variants (complete)", "whisper-small -> Qwen, noisy", "base_noisy.jsonl", 120, 8, True),
 ("Baseline + first variants (complete)", "gold-text oracle", "oracle.jsonl", 120, 3, True),
 ("Baseline + first variants (complete)", "NLLB-grounded (small), clean+noisy", "base+mt_noisy.jsonl", 120, 18, True),
 ("Baseline + first variants (complete)", "turbo ASR, clean+noisy", "asr-turbo_noisy.jsonl", 120, 8, True),
 ("Baseline + first variants (complete)", "turbo -> Qwen, clean+noisy", "turbo_noisy.jsonl", 120, 12, True),
 ("Baseline + first variants (complete)", "turbo + NLLB + Qwen, clean+noisy", "turbo+mt_noisy.jsonl", 120, 16, True),
 ("Baseline + first variants (complete)", "NLLB-only controls", "mtonly-turbo_noisy.jsonl", 120, 15, True),
 ("IndicConformer on the translation set", "Conformer RNNT ASR, clean", "asr-conf-rnnt_clean.jsonl", 120, 1.5, True),
 ("IndicConformer on the translation set", "Conformer RNNT ASR, noisy", "asr-conf-rnnt_noisy.jsonl", 120, 1.5, True),
 ("IndicConformer on the translation set", "Conformer CTC ASR, clean", "asr-conf-ctc_clean.jsonl", 120, 1.5, True),
 ("IndicConformer on the translation set", "Conformer CTC ASR, noisy", "asr-conf-ctc_noisy.jsonl", 120, 1.5, True),
 ("Listening comprehension (Belebele-Fleurs, 200 questions)", "transcribe passages: gold", "qa/asr_gold.jsonl", 1, 0.2, False),
 ("Listening comprehension (Belebele-Fleurs, 200 questions)", "transcribe passages: whisper-small", "qa/asr_small.jsonl", 1, 12, False),
 ("Listening comprehension (Belebele-Fleurs, 200 questions)", "transcribe passages: whisper-turbo", "qa/asr_turbo.jsonl", 1, 9, False),
 ("Listening comprehension (Belebele-Fleurs, 200 questions)", "transcribe passages: IndicConformer", "qa/asr_conf.jsonl", 1, 7, False),
 ("Listening comprehension (Belebele-Fleurs, 200 questions)", "NLLB translate passages + questions", "qa/mt_conf.jsonl", 1, 6, False),
] + [("Listening comprehension (Belebele-Fleurs, 200 questions)", f"Qwen answers: {n}, {m}", f"qa/ans_{t}_{m}.jsonl", 200, 4, True)
     for t, n in [("gold", "gold text"), ("small", "whisper-small"), ("turbo", "whisper-turbo"), ("conf", "Conformer")] for m in ("native", "en")] + [
 ("IndicConformer on the translation set (LLM stages)", "Conformer -> Qwen, clean", "conf_clean.jsonl", 120, 4, True),
 ("IndicConformer on the translation set (LLM stages)", "Conformer -> Qwen, noisy", "conf_noisy.jsonl", 120, 4, True),
 ("IndicConformer on the translation set (LLM stages)", "Conformer + NLLB + Qwen, clean", "conf+mt_clean.jsonl", 120, 6, True),
 ("IndicConformer on the translation set (LLM stages)", "Conformer + NLLB + Qwen, noisy", "conf+mt_noisy.jsonl", 120, 6, True),
 ("IndicConformer on the translation set (LLM stages)", "Conformer -> NLLB only, clean", "mtonly-conf_clean.jsonl", 120, 2, True),
 ("IndicConformer on the translation set (LLM stages)", "Conformer -> NLLB only, noisy", "mtonly-conf_noisy.jsonl", 120, 2, True),
]
# which log tqdm label belongs to which stage file (only needed for non-incremental stages)
DESC = {"qa/asr_small.jsonl": "asr/small", "qa/asr_turbo.jsonl": "asr/turbo", "qa/asr_conf.jsonl": "asr/conf"}
W = 26


def rows(f):
    p = os.path.join(R, f)
    if not os.path.exists(p): return 0
    with open(p, "rb") as fh: return sum(1 for _ in fh)


def log_now():
    try:
        with open(LOG, "rb") as fh:
            fh.seek(max(0, os.path.getsize(LOG) - 6000)); t = fh.read().decode("utf-8", "ignore").replace("\r", "\n")
        m = re.findall(r"([\w+\-/]+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", t)
        return (m[-1][0], int(m[-1][1]), int(m[-1][2])) if m else None
    except OSError: return None


def etime():
    out = subprocess.run(["ps", "-axo", "etime=,command="], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if ("run_qa_and_conformer" in l or "run_rest" in l) and "ps -axo" not in l:
            p = l.split()[0].replace("-", ":").split(":")[::-1]; s = sum(int(x) * m for x, m in zip(p, (1, 60, 3600, 86400))); return s
    return None


def bar(f, w=W): n = int(w * f); return "\033[32m" + "█" * n + "\033[90m" + "░" * (w - n) + "\033[0m"


try:
    while True:
        now = log_now(); frac = []
        for g, lab, f, n, est, inc in S:
            r = rows(f)
            if inc: fr = min(1.0, r / n)
            else:
                fr = 1.0 if os.path.exists(os.path.join(R, f)) else 0.0
                if fr == 0 and now and DESC.get(f) == now[0]: fr = now[1] / now[2]
            frac.append(fr)
        tot = sum(e for *_, e, _ in S); done = sum(fr * e for fr, (_, _, _, _, e, _) in zip(frac, S))
        el = etime(); running = el is not None
        eta = ""
        if running and done > 0:
            # scale the estimates by how long the *current run's* finished work actually took
            run_done = sum(fr * e for fr, (g, *_ , e, _) in zip(frac, S) if not g.startswith("Baseline"))
            run_tot = sum(e for (g, *_ , e, _) in S if not g.startswith("Baseline"))
            if run_done > 1: eta = f"  ETA ~{max(0,(run_tot-run_done)*(el/60)/run_done):.0f} min"
        print("\033[2J\033[H", end="")
        print(("\033[1;32mRUNNING\033[0m" if running else "\033[1;33mnot running\033[0m (finished or stopped)") +
              (f"   elapsed {el//3600}h {el%3600//60:02d}m" if running else ""))
        if now: print(f"now: {now[0]}  {now[1]}/{now[2]}")
        last = None
        for fr, (g, lab, f, n, est, inc) in zip(frac, S):
            if g != last: print(f"\n\033[1m{g}\033[0m"); last = g
            mark = "\033[32mdone\033[0m" if fr >= 1 else (f"{100*fr:3.0f}%" if fr > 0 else "")
            print(f"  {lab:<40} [{bar(fr)}] {mark}")
        rd = sum(fr * e for fr, (g, *_, e, _) in zip(frac, S) if not g.startswith("Baseline"))
        rt = sum(e for (g, *_, e, _) in S if not g.startswith("Baseline"))
        print(f"\nTHIS RUN      [{bar(rd/rt, 40)}] {100*rd/rt:.0f}%{eta}")
        print(f"WHOLE PROJECT [{bar(done/tot, 40)}] {100*done/tot:.0f}%  (includes the earlier completed work)")
        print("\nStages that write their file at the end show 0% until done (the 'now:' line shows live progress).")
        if done >= tot - 1e-6: print("\033[1;32mALL STAGES DONE\033[0m"); break
        time.sleep(3)
except KeyboardInterrupt:
    pass
