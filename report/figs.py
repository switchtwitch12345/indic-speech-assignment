"""Hand-built SVG architecture diagrams (no external deps)."""
C = dict(enc="#dbeafe", llm="#e9d5ff", dec="#fde68a", tok="#d1fae5", io="#f3f4f6", ext="#fee2e2")


def svg(w, h, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'style="width:100%;max-width:{w}px;font-family:Helvetica,Arial,sans-serif;font-size:11px">'
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#374151"/></marker></defs>' + body + "</svg>")


def box(x, y, w, h, text, fill, bold=False, dash=False):
    lines = text.split("\n"); ty = y + h / 2 - (len(lines) - 1) * 6.5 + 4
    t = "".join(f'<text x="{x+w/2}" y="{ty+i*13}" text-anchor="middle"{" font-weight=700" if bold and i==0 else ""}>{l}</text>'
                for i, l in enumerate(lines))
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#374151"{d}/>' + t


def arrow(x1, y1, x2, y2, label="", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="1.4" marker-end="url(#a)"{d}/>'
    if label:
        s += f'<text x="{(x1+x2)/2+4}" y="{(y1+y2)/2-4}" fill="#4b5563" font-size="10">{label}</text>'
    return s


def txt(x, y, s, size=11, anchor="start", color="#111827", bold=False):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}"{" font-weight=700" if bold else ""}>{s}</text>'


def moshi():
    b = txt(10, 16, "Moshi: one 7B Temporal Transformer, two audio streams in, two token streams out, 12.5 Hz frames", 12, bold=True)
    b += box(10, 40, 110, 46, "User speech\n24 kHz (always on)", C["io"])
    b += box(150, 40, 140, 46, "Mimi encoder\n(causal, streaming)", C["enc"])
    b += box(320, 22, 140, 38, "1 semantic token\n(WavLM-distilled)", C["tok"])
    b += box(320, 66, 140, 38, "7 acoustic RVQ tokens", C["tok"])
    b += arrow(120, 63, 150, 63) + arrow(290, 55, 320, 41) + arrow(290, 70, 320, 85)
    b += box(500, 40, 180, 64, "Temporal Transformer (Helium 7B)\ninput = SUM of learned embeddings\nof user + Moshi streams + text", C["llm"], bold=True)
    b += arrow(460, 41, 500, 58) + arrow(460, 85, 500, 86)
    b += box(500, 140, 180, 44, "Text 'inner monologue'\nPAD / EPAD time-aligned words", C["tok"])
    b += arrow(590, 104, 590, 140, "predicts first")
    b += box(500, 218, 180, 44, "Depth Transformer (6 layers, d=1024)\nautoregressive over 8 codebooks", C["dec"])
    b += arrow(590, 184, 590, 218)
    b += box(320, 218, 140, 44, "Moshi tokens\n(1 semantic + 7 acoustic)\nacoustic delay = 1 frame", C["tok"])
    b += arrow(500, 240, 460, 240)
    b += box(150, 218, 140, 44, "Mimi decoder\n(streaming)", C["enc"]) + arrow(320, 240, 290, 240)
    b += box(10, 218, 110, 44, "Moshi speech\n(speech or silence)", C["io"]) + arrow(150, 240, 120, 240)
    b += txt(10, 300, "Latency floor: 80 ms frame + 80 ms acoustic delay = 160 ms theoretical, about 200 ms measured. No end-of-turn token: the model", 10, color="#4b5563")
    b += txt(10, 313, "always emits audio; 'not speaking' = near-silent waveform + PAD text. Forcing EPAD makes it start talking.", 10, color="#4b5563")
    return svg(700, 325, b)


def qwen():
    b = txt(10, 16, "Qwen3-Omni: Thinker (understands, writes text) and Talker (speaks), streamed in chunks", 12, bold=True)
    b += box(10, 40, 100, 44, "User audio\n(128-ch mel)", C["io"])
    b += box(135, 36, 140, 52, "AuT encoder ~0.6B\nattn enc-dec, 12.5 Hz\nwindow 1-8 s", C["enc"])
    b += arrow(110, 62, 135, 62)
    b += box(300, 28, 150, 68, "Thinker  MoE 30B-A3B\nTM-RoPE: 80 ms time IDs\nchunked prefill\n(text / image / video / audio in)", C["llm"], bold=True)
    b += arrow(275, 62, 300, 62)
    b += box(485, 28, 100, 68, "Text reply\n(streamed tokens)", C["tok"]) + arrow(450, 62, 485, 62)
    b += box(300, 140, 150, 56, "Talker  MoE 3B-A0.3B\nconditions on audio/visual feats +\nthe turn's streamed text", C["llm"], bold=True)
    b += arrow(375, 96, 375, 140)
    b += arrow(535, 96, 450, 160, dash=True)
    b += box(485, 140, 100, 56, "MTP ~80M\nresidual codebooks\nper frame", C["dec"]) + arrow(450, 168, 485, 168)
    b += box(280, 232, 190, 52, "Code2Wav ~200M\ncausal ConvNet (non-AR)\nstreams from 1st codec frame", C["dec"])
    b += arrow(535, 196, 420, 232)
    b += box(500, 232, 100, 52, "Speech reply\n(12.5 Hz codec)", C["io"]) + arrow(470, 258, 500, 258)
    b += box(10, 140, 150, 56, "External (not in model):\nVAD / endpointing, RAG,\nfunction calls, safety filters", C["ext"], dash=True)
    b += arrow(160, 160, 300, 70, dash=True)
    b += txt(10, 305, "First-packet latency (paper, theoretical): 234 ms audio / 547 ms video at 1 stream; 728 ms at 4 and 1172 ms at 6 concurrent streams.", 10, color="#4b5563")
    b += txt(10, 318, "Speech input languages (19) do not include Hindi or Tamil; AuT pre-training data was 80% Chinese/English ASR, 10% other-language ASR.", 10, color="#4b5563")
    return svg(700, 330, b)


def proposal():
    b = txt(10, 16, "Proposed: Speculative Think-while-Listening (SpecTL) with a learned turn-end head", 12, bold=True)
    b += box(10, 40, 130, 56, "Streaming speech\n12.5 Hz encoder states", C["io"])
    b += box(170, 40, 150, 56, "Causal speech encoder\n(multilingual,\nIndic-balanced)", C["enc"]) + arrow(140, 68, 170, 68)
    b += box(360, 22, 200, 56, "Turn-end head  p(end | prefix)\nsemantic + prosody cues,\nlatency-aware loss", C["ext"], bold=True)
    b += box(360, 92, 200, 56, "Thinker drafts a reply while\nthe user is still speaking\n(KV-cache snapshots)", C["llm"], bold=True)
    b += arrow(320, 56, 360, 46) + arrow(320, 80, 360, 112) + arrow(460, 78, 460, 92, "gate")
    b += box(600, 22, 150, 56, "p > theta_hi: commit\np < theta_lo: roll back\nto the snapshot", C["dec"])
    b += arrow(560, 50, 600, 50)
    b += box(600, 92, 150, 56, "Talker streams audio\nfrom the committed\ndraft", C["tok"]) + arrow(560, 120, 600, 120) + arrow(675, 78, 675, 92)
    b += txt(10, 178, "Training signal: (i) turn-end labels from word-aligned dialogue audio incl. Hindi/Tamil/code-mixed;", 10, color="#4b5563")
    b += txt(10, 191, "(ii) loss = BCE + lambda * (time between predicted and true end of turn); (iii) rollback cost = wasted Thinker tokens.", 10, color="#4b5563")
    return svg(760, 205, b)
