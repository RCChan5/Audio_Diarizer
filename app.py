"""
Audio Diarizer — Batch Edition (accuracy-tuned)
• Drag-and-drop a folder of AMR/MP3/WAV/M4A files
• Transcribes each with faster-whisper (large-v3-turbo, int8_float32)
• Diarizes speakers with pyannote.audio 3.1
• Matches the SAME PERSON across files using voice embeddings
• Outputs one .txt transcript per audio file
"""

import os, sys, shutil, tempfile, subprocess, warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Make FFmpeg shared DLLs discoverable for torchcodec (Windows / pyannote 4.x)
# ──────────────────────────────────────────────────────────────────────────────
_FFMPEG_SHARED_BIN = r"C:\Users\topsn\ffmpeg-shared\ffmpeg-8.1.1-full_build-shared\bin"
if os.name == "nt" and os.path.isdir(_FFMPEG_SHARED_BIN):
    try:
        os.add_dll_directory(_FFMPEG_SHARED_BIN)
    except Exception:
        pass
    os.environ["PATH"] = _FFMPEG_SHARED_BIN + os.pathsep + os.environ.get("PATH", "")

FFMPEG_EXE = "ffmpeg"
for _candidate in (
    os.path.join(_FFMPEG_SHARED_BIN, "ffmpeg.exe"),
    "ffmpeg",
):
    if os.path.isfile(_candidate) or shutil.which(_candidate):
        FFMPEG_EXE = _candidate
        break

# ──────────────────────────────────────────────────────────────────────────────
# Dependency bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for _mod, _pkg in {
    "gradio":         "gradio>=4.0",
    "faster_whisper": "faster-whisper",
    "torch":          "torch",
    "torchaudio":     "torchaudio",
    "pyannote":       "pyannote.audio",
    "numpy":          "numpy",
}.items():
    try:
        __import__(_mod)
    except ImportError:
        print(f"Installing {_pkg}…")
        _install(_pkg)

import gradio as gr
import torch
import numpy as np
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────────────────────────────────────
# Globals — models loaded once and reused
# ──────────────────────────────────────────────────────────────────────────────
_whisper_model  = None
_whisper_key    = None   # (model_size,) — reload if user changes size
_diar_pipeline  = None
_embed_pipeline = None

SUPPORTED = {".amr", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".wma"}

# CPU threads — Ryzen 3900X has 12 physical cores
CPU_THREADS = max(1, (os.cpu_count() or 12))

# Load HuggingFace token from api_keys.txt (KEY=VALUE per line)
def _load_api_keys(path: Path) -> dict:
    keys = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys

_API_KEYS = _load_api_keys(Path(__file__).parent / "api_keys.txt")
HF_TOKEN = _API_KEYS.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")

# Expose token to huggingface_hub so model downloads are authenticated
# (silences "unauthenticated requests" warning and lifts rate limits)
os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

# Default output folder
DEFAULT_OUTPUT = r"C:\Users\topsn\Desktop\diarization\output"


# ──────────────────────────────────────────────────────────────────────────────
# Audio preprocessing — ffmpeg denoise + normalize + 16 kHz mono
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_audio(src: Path, dst: Path):
    """
    Convert any audio format to a clean 16-kHz mono WAV with light denoise +
    loudness normalization. Helps WER significantly on phone/AMR recordings.
    """
    cmd = [
        FFMPEG_EXE, "-y", "-i", str(src),
        "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy model loaders
# ──────────────────────────────────────────────────────────────────────────────
def get_whisper(model_size: str):
    """
    faster-whisper on CPU with int8_float32 — accuracy ≈ fp16 but fast on CPU.
    `large-v3-turbo` is the accuracy/speed sweet spot for a 12-core Ryzen.
    """
    global _whisper_model, _whisper_key
    if _whisper_model is None or _whisper_key != model_size:
        print(f"  Loading faster-whisper '{model_size}' (int8_float32, {CPU_THREADS} threads)…")
        _whisper_model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8_float32",
            cpu_threads=CPU_THREADS,
            num_workers=1,
        )
        _whisper_key = model_size
    return _whisper_model


def get_diar_pipeline(hf_token: str):
    global _diar_pipeline
    if _diar_pipeline is None:
        from pyannote.audio import Pipeline
        print("  Loading pyannote diarization pipeline…")
        _diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        if torch.cuda.is_available():
            _diar_pipeline = _diar_pipeline.to(torch.device("cuda"))
    return _diar_pipeline


def get_embed_pipeline(hf_token: str):
    global _embed_pipeline
    if _embed_pipeline is None:
        from pyannote.audio import Inference
        print("  Loading pyannote embedding model…")
        _embed_pipeline = Inference(
            "pyannote/embedding",
            window="whole",
            token=hf_token,
        )
        if torch.cuda.is_available():
            _embed_pipeline = _embed_pipeline.to(torch.device("cuda"))
    return _embed_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────
def transcribe(wav_path: Path, model_size: str, language: str | None) -> list:
    """
    faster-whisper transcription with accuracy-tuned settings:
      • beam_size=5            — better decoding than greedy
      • vad_filter=True        — skip silence (kills hallucinations)
      • condition_on_previous_text=False — prevents error cascades on long audio
      • explicit language      — avoids misdetection on short/noisy clips
    """
    model = get_whisper(model_size)
    segments_iter, _info = model.transcribe(
        str(wav_path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        temperature=0.0,
    )
    out = []
    for s in segments_iter:
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text})
    return out


def diarize(wav_path: Path, hf_token: str, num_speakers=None):
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = get_diar_pipeline(hf_token)(str(wav_path), **kwargs)
    # pyannote 4.x wraps the Annotation in a DiarizeOutput dataclass
    return getattr(result, "speaker_diarization", result)


def extract_speaker_embeddings(wav_path: Path, diarization, hf_token: str) -> dict:
    """
    Build a voice-fingerprint (embedding) for every speaker in this file.
    Pass tensors directly to Inference instead of round-tripping through temp WAVs.
    """
    import torchaudio

    inference = get_embed_pipeline(hf_token)
    waveform, sr = torchaudio.load(str(wav_path))

    speaker_vecs = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end - turn.start < 0.5:
            continue
        s = int(turn.start * sr)
        e = int(turn.end   * sr)
        chunk = waveform[:, s:e]
        if chunk.shape[1] < sr * 0.5:
            continue
        try:
            emb = inference({"waveform": chunk, "sample_rate": sr})
            speaker_vecs[speaker].append(np.array(emb))
        except Exception:
            pass

    return {
        spk: np.mean(vecs, axis=0)
        for spk, vecs in speaker_vecs.items()
        if vecs
    }


def assign_segments(whisper_segs: list, diarization) -> list:
    """Map each Whisper segment to the speaker with the most time overlap."""
    enriched = []
    for seg in whisper_segs:
        start, end   = seg["start"], seg["end"]
        best_spk     = "UNKNOWN"
        best_overlap = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            ov = max(0.0, min(end, turn.end) - max(start, turn.start))
            if ov > best_overlap:
                best_overlap = ov
                best_spk     = speaker
        enriched.append({
            "speaker": best_spk,
            "start":   round(start, 2),
            "end":     round(end,   2),
            "text":    seg["text"].strip(),
        })
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Cross-file speaker identity matching
# ──────────────────────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def build_global_speaker_map(all_embeddings: list, threshold: float = 0.75) -> list:
    registry: list[tuple[str, np.ndarray]] = []
    maps = []

    for file_embs in all_embeddings:
        local_map = {}
        for local_lbl, emb in file_embs.items():
            best_score, best_idx = -1.0, None
            for idx, (_, g_emb) in enumerate(registry):
                score = cosine_sim(emb, g_emb)
                if score > best_score:
                    best_score, best_idx = score, idx

            if best_idx is not None and best_score >= threshold:
                g_lbl, old_emb = registry[best_idx]
                registry[best_idx] = (g_lbl, (old_emb + emb) / 2)
                local_map[local_lbl] = g_lbl
            else:
                new_lbl = f"PERSON_{chr(65 + len(registry))}"
                registry.append((new_lbl, emb))
                local_map[local_lbl] = new_lbl

        maps.append(local_map)

    return maps


def remap_speakers(segments: list, speaker_map: dict) -> list:
    return [{**s, "speaker": speaker_map.get(s["speaker"], s["speaker"])} for s in segments]


# ──────────────────────────────────────────────────────────────────────────────
# Text output formatting
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_txt(filename: str, segments: list, speaker_map: dict) -> str:
    lines = [f"TRANSCRIPT: {filename}", "=" * 60, ""]

    if speaker_map:
        lines.append("SPEAKER LEGEND (cross-file consistent labels)")
        for local, gbl in sorted(speaker_map.items()):
            lines.append(f"  {local:20s} → {gbl}")
        lines += ["", "-" * 60, ""]

    current_spk = None
    for seg in segments:
        spk = seg["speaker"]
        if spk != current_spk:
            lines.append(f"\n[{spk}]")
            current_spk = spk
        lines.append(f"  [{_fmt(seg['start'])} → {_fmt(seg['end'])}] {seg['text']}")

    lines += ["", "=" * 60, "SPEAKER SUMMARY", ""]
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int]   = defaultdict(int)
    for seg in segments:
        totals[seg["speaker"]] += seg["end"] - seg["start"]
        counts[seg["speaker"]] += 1
    total_dur = sum(totals.values()) or 1
    for spk in sorted(totals):
        pct  = totals[spk] / total_dur * 100
        m, s = divmod(int(totals[spk]), 60)
        lines.append(f"  {spk}: {m}m {s}s ({pct:.1f}%)  —  {counts[spk]} segments")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Gradio handler
# ──────────────────────────────────────────────────────────────────────────────
def process_folder(files, hf_token, model_size, num_speakers_str,
                   sim_threshold, output_folder, language):

    if not files:
        yield "❌ No files uploaded.", ""
        return
    token = (hf_token or "").strip() or HF_TOKEN
    if not token:
        yield "❌ HuggingFace token required.", ""
        return

    num_speakers = int(num_speakers_str.strip()) if num_speakers_str.strip().isdigit() else None
    out_dir      = Path(output_folder.strip()) if output_folder.strip() else Path(DEFAULT_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir      = Path(tempfile.mkdtemp())
    lang_code    = (language or "").strip().lower() or None

    audio_files = []
    for f in files:
        p = Path(f.name if hasattr(f, "name") else str(f))
        if p.suffix.lower() in SUPPORTED:
            audio_files.append(p)

    if not audio_files:
        yield f"❌ No supported audio files found ({', '.join(sorted(SUPPORTED))}).", ""
        return

    total = len(audio_files)
    log   = []

    def push(msg):
        log.append(msg)
        return "\n".join(log)

    yield push(
        f"📁 {total} audio file(s) found.\n"
        f"⚙️  Whisper: {model_size} (int8_float32, {CPU_THREADS} threads) · "
        f"language={lang_code or 'auto'}\n"
    ), ""

    # ── Phase 1: preprocess → transcribe → diarize ───────────────────────────
    all_data: list[dict] = []

    for i, src in enumerate(audio_files, 1):
        yield push(f"[{i}/{total}] {src.name}  — preprocessing (denoise + normalize)…"), ""
        wav = tmp_dir / f"{i:03d}_{src.stem}.wav"
        try:
            preprocess_audio(src, wav)
        except Exception as e:
            yield push(f"  ⚠️  Preprocessing failed: {e}"), ""
            continue

        yield push(f"[{i}/{total}] {src.name}  — transcribing…"), ""
        try:
            segs = transcribe(wav, model_size, lang_code)
        except Exception as e:
            yield push(f"  ⚠️  Transcription failed: {e}"), ""
            continue

        yield push(f"[{i}/{total}] {src.name}  — diarizing speakers…"), ""
        try:
            diarization = diarize(wav, token, num_speakers)
            enriched    = assign_segments(segs, diarization)
        except Exception as e:
            yield push(f"  ⚠️  Diarization failed: {e}"), ""
            continue

        all_data.append({"src": src, "wav": wav, "segments": enriched, "diarization": diarization})
        yield push(f"  ✅ {len(enriched)} segments extracted"), ""

    if not all_data:
        yield push("\n❌ All files failed."), ""
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # ── Phase 2: voice embeddings ────────────────────────────────────────────
    yield push("\n🔬 Extracting voice fingerprints for cross-file identity matching…"), ""
    all_embeddings = []
    embed_ok = True
    for item in all_data:
        try:
            embs = extract_speaker_embeddings(item["wav"], item["diarization"], token)
            all_embeddings.append(embs)
            yield push(f"  🎤 {item['src'].name}: {len(embs)} speaker(s) fingerprinted"), ""
        except Exception as e:
            yield push(f"  ⚠️  Embedding skipped for {item['src'].name}: {e}"), ""
            all_embeddings.append({})
            embed_ok = False

    # ── Phase 3: global identity map ─────────────────────────────────────────
    if embed_ok and any(all_embeddings):
        yield push(f"\n🧠 Matching voices across files (threshold = {sim_threshold:.2f})…"), ""
        speaker_maps = build_global_speaker_map(all_embeddings, threshold=sim_threshold)
    else:
        yield push("⚠️  Cross-file matching skipped."), ""
        speaker_maps = [{} for _ in all_data]

    # ── Phase 4: write TXT transcripts ───────────────────────────────────────
    yield push("\n💾 Writing transcript files…"), ""
    written = []
    for item, spk_map in zip(all_data, speaker_maps):
        final_segs = remap_speakers(item["segments"], spk_map)
        txt_name   = item["src"].stem + "_transcript.txt"
        txt_path   = out_dir / txt_name
        txt_path.write_text(
            format_txt(item["src"].name, final_segs, spk_map),
            encoding="utf-8"
        )
        written.append(str(txt_path))
        yield push(f"  ✅ {txt_name}"), ""

    # ── Phase 5: master summary ──────────────────────────────────────────────
    summary_path = out_dir / "_batch_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("BATCH DIARIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for item, spk_map in zip(all_data, speaker_maps):
            f.write(f"File: {item['src'].name}\n")
            for loc, gbl in sorted(spk_map.items()):
                f.write(f"  {loc:20s} → {gbl}\n")
            f.write("\n")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    final = push(
        f"\n✅ Done! {len(written)} / {total} files processed.\n"
        f"📂 Output folder: {out_dir}"
    )
    file_list = "\n".join(written + [str(summary_path)])
    yield final, file_list


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root{--bg:#0d0d0f;--panel:#141418;--border:#2a2a32;--accent:#7c6af7;--accent2:#f76a8e;
      --text:#e8e8f0;--muted:#7070a0;--success:#4ade80;
      --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;}
body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
.gradio-container{max-width:1140px!important;margin:0 auto!important;}
#hdr{padding:2.5rem 0 1.2rem;text-align:center;border-bottom:1px solid var(--border);margin-bottom:1.8rem;}
#hdr h1{font-size:2.3rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .3rem;letter-spacing:-1px;}
#hdr p{color:var(--muted);font-family:var(--mono);font-size:.75rem;margin:0;}
.panel{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1.1rem!important;}
label{font-family:var(--mono)!important;font-size:.72rem!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:1px!important;}
input,textarea,select{background:#1a1a22!important;border:1px solid var(--border)!important;
    color:var(--text)!important;border-radius:8px!important;font-family:var(--mono)!important;}
button.primary{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;border:none!important;
    color:#fff!important;font-family:var(--mono)!important;font-weight:700!important;
    font-size:.85rem!important;padding:.8rem 2rem!important;border-radius:8px!important;}
.callout{background:rgba(124,106,247,.07);border:1px solid rgba(124,106,247,.28);border-radius:8px;
    padding:.85rem 1rem;font-family:var(--mono);font-size:.71rem;color:#aaa8e8;line-height:1.75;margin-bottom:1rem;}
.log textarea{font-family:var(--mono)!important;font-size:.75rem!important;color:var(--success)!important;min-height:300px!important;}
.files textarea{font-family:var(--mono)!important;font-size:.72rem!important;min-height:110px!important;}
"""

def build_ui():
    with gr.Blocks(css=CSS, title="Audio Diarizer — Batch") as demo:
        gr.HTML("""
        <div id="hdr">
          <h1>🎙 Audio Diarizer</h1>
          <p>batch amr/mp3/wav · faster-whisper · cross-file speaker identity · txt output</p>
        </div>""")

        gr.HTML("""
        <div class="callout">
          <strong>Before first run:</strong> &nbsp;
          Get a free token → <a href="https://huggingface.co/settings/tokens" target="_blank" style="color:#7c6af7">huggingface.co/settings/tokens</a>
          &nbsp;|&nbsp; Accept licences for
          <a href="https://hf.co/pyannote/speaker-diarization-3.1" target="_blank" style="color:#7c6af7">speaker-diarization-3.1</a>,
          <a href="https://hf.co/pyannote/segmentation-3.0" target="_blank" style="color:#7c6af7">segmentation-3.0</a>, and
          <a href="https://hf.co/pyannote/embedding" target="_blank" style="color:#7c6af7">embedding</a>
          &nbsp;|&nbsp; First run downloads ~1.5 GB (cached afterwards)<br>
          <strong>Usage:</strong> Select your audio files and drag them into the box, or click to browse.
        </div>""")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Audio Files (AMR, MP3, WAV, M4A, FLAC…)",
                    file_count="multiple",
                    elem_classes=["panel"],
                )
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    value=HF_TOKEN,
                    type="password",
                    elem_classes=["panel"],
                )
                output_folder = gr.Textbox(
                    label="Output folder",
                    value=DEFAULT_OUTPUT,
                    placeholder=r"e.g.  C:\Users\You\transcripts",
                    elem_classes=["panel"],
                )
                with gr.Row():
                    model_size = gr.Dropdown(
                        label="Whisper Model",
                        choices=["tiny","base","small","medium","large-v3","large-v3-turbo"],
                        value="large-v3-turbo",
                    )
                    num_speakers = gr.Textbox(
                        label="# Speakers (optional)",
                        placeholder="e.g. 2",
                    )
                language = gr.Textbox(
                    label="Language code (en, es, fr… blank = auto)",
                    value="en",
                    placeholder="en",
                )
                sim_threshold = gr.Slider(
                    label="Cross-file voice match threshold",
                    minimum=0.50, maximum=0.95, value=0.75, step=0.01,
                    info="Higher = stricter (same person must sound more alike to be merged)",
                )
                run_btn = gr.Button("▶  Process All Files", variant="primary")

            with gr.Column(scale=2):
                log_out = gr.Textbox(
                    label="Progress Log",
                    interactive=False,
                    elem_classes=["log","panel"],
                )
                files_out = gr.Textbox(
                    label="Files Written",
                    interactive=False,
                    elem_classes=["files","panel"],
                )

        run_btn.click(
            fn=process_folder,
            inputs=[file_input, hf_token, model_size, num_speakers,
                    sim_threshold, output_folder, language],
            outputs=[log_out, files_out],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860,
                      share=False, show_error=True)
