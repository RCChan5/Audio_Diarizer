# Audio Diarizer — Batch Transcription with Cross-File Speaker Identity

> An end-to-end, fully-local speech analytics pipeline that transcribes batches of audio recordings, separates speakers within each file, and matches the **same person across multiple files** using voice fingerprinting.

Built to solve a real problem: when you have dozens of phone recordings, interviews, or meetings, generic transcription tools label speakers as `SPEAKER_00`, `SPEAKER_01` — and those labels reset every file. This project produces consistent identities (`PERSON_A`, `PERSON_B`, …) **across the entire batch**, so the same voice gets the same label no matter which file it appears in.

---

## Highlights

- **Batch processing** — drag in a folder of mixed-format recordings (AMR, MP3, WAV, M4A, FLAC, OGG, MP4, WMA) and walk away.
- **Cross-file speaker identity matching** — voice embeddings + cosine similarity register each speaker once in a global registry, so "PERSON_A" in file 3 is the same person as "PERSON_A" in file 17.
- **Accuracy-tuned ASR** — `faster-whisper` (CTranslate2 backend) with `large-v3-turbo`, beam search, VAD filtering, and `condition_on_previous_text=False` to prevent hallucination cascades on long audio.
- **Audio preprocessing pipeline** — ffmpeg chain (`highpass → lowpass → afftdn denoise → loudnorm → 16 kHz mono`) measurably improves WER on noisy phone/AMR sources.
- **CPU-first, GPU-optional** — runs entirely on a 12-core Ryzen using `int8_float32` quantization; auto-detects and uses CUDA if present.
- **100% local & private** — audio never leaves the machine. The only network call is the one-time model weight download.
- **Polished Gradio UI** — custom dark theme, live progress log, no Python knowledge needed to operate.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Transcription | `faster-whisper` (large-v3-turbo, int8_float32) | ~4× CPU throughput vs. reference Whisper, near-identical WER |
| Diarization | `pyannote.audio` 3.1 speaker-diarization pipeline | SOTA open-source diarization |
| Speaker embeddings | `pyannote/embedding` | Voice fingerprints for cross-file identity |
| Audio I/O | `ffmpeg` (subprocess) + `torchaudio` | Format-agnostic ingest, in-memory tensors to avoid temp-file thrash |
| ML runtime | `PyTorch` 2.x | CUDA-aware, CPU-fallback |
| UI | `Gradio` 4.x | Single-file deploy, custom CSS theming |
| Language | Python 3.9+ | — |

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Batch of    │ ─▶ │   ffmpeg     │ ─▶ │ faster-whisper  │ ─▶ │   Per-segment    │
│  audio files │    │  preprocess  │    │  transcription  │    │      text        │
└──────────────┘    └──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                                        │
                           ┌────────────────────────────────────────────┘
                           ▼
                   ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
                   │   pyannote   │ ─▶ │ Per-file speaker │ ─▶ │  Segment-speaker │
                   │  diarization │    │     turns        │    │     alignment    │
                   └──────────────┘    └────────┬─────────┘    └────────┬─────────┘
                                                │                       │
                                                ▼                       │
                                       ┌──────────────────┐             │
                                       │  Voice embedding │             │
                                       │   per speaker    │             │
                                       └────────┬─────────┘             │
                                                ▼                       │
                                       ┌──────────────────┐             │
                                       │ Global registry  │             │
                                       │ (cosine match)   │             │
                                       └────────┬─────────┘             │
                                                ▼                       ▼
                                          ┌──────────────────────────────────┐
                                          │  TXT transcripts + batch summary │
                                          └──────────────────────────────────┘
```

### Cross-file matching algorithm

For each file, an embedding is computed per speaker turn (>0.5 s) and averaged into a single per-speaker vector. Vectors are matched against a running global registry using cosine similarity; if the best match exceeds a tunable threshold (default `0.75`), the local label is mapped to the existing global identity and the registry vector is updated as a running mean. Otherwise a new global identity is registered. The result is a consistent `PERSON_A / PERSON_B / …` namespace across the entire batch.

---

## Engineering Decisions Worth Calling Out

- **`int8_float32` quantization on CPU** — keeps accuracy within margin-of-noise of fp16 while making `large-v3-turbo` viable on a 12-core CPU without a GPU.
- **VAD filtering + `condition_on_previous_text=False`** — eliminates two of the most common Whisper failure modes on long noisy audio: silence hallucinations and error cascades where a bad early decode poisons the rest of the file.
- **In-memory tensor pass to the embedding model** — avoids round-tripping per-speaker chunks through temporary WAV files, cutting per-file I/O substantially on batches with many speaker turns.
- **Running-mean registry update** — every time a global identity is matched, its embedding is re-averaged with the new evidence, so the registry drifts toward the true centroid of a person's voice as more samples arrive.
- **Lazy, cached model loaders** — Whisper / diarization / embedding pipelines are loaded once per process and keyed by model size, so flipping between runs in the UI doesn't re-pay the multi-second warmup.

---

## Output Format

Per file, a `*_transcript.txt`:

```
TRANSCRIPT: call.amr
============================================================

SPEAKER LEGEND (cross-file consistent labels)
  SPEAKER_00           → PERSON_A
  SPEAKER_01           → PERSON_C

------------------------------------------------------------

[PERSON_A]
  [00:03 → 00:08]  Hello is this John?
  [00:09 → 00:15]  We’ve been trying to reach you about your car’s extended warranty.

[PERSON_C]
  [00:16 → 00:22]  Sorry wrong number.

============================================================
SPEAKER SUMMARY

  PERSON_A: 4m 12s (62.3%)  —  18 segments
  PERSON_C: 2m 33s (37.7%)  —  11 segments
```

Plus a top-level `_batch_summary.txt` mapping every local label in every file to its global identity.

---

## Quick Start

### Prerequisites
- Python 3.9+
- `ffmpeg` on PATH (`brew install ffmpeg` / `apt install ffmpeg` / [ffmpeg.org](https://ffmpeg.org/download.html))
- A free HuggingFace account + accepted licences for [`speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1), [`segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0), and [`embedding`](https://hf.co/pyannote/embedding)

### Setup
```bash
pip install -r requirements.txt
```

Create `api_keys.txt` in the project root:
```
HF_TOKEN=hf_your_token_here
```
(`api_keys.txt` is gitignored.)

### Run
```bash
python app.py
```
Open http://localhost:7860, drop in audio files, click **Process All Files**.

---

## Configuration Knobs

| UI Field | Effect |
|---|---|
| Whisper Model | `tiny` → `large-v3-turbo`. Default `large-v3-turbo` (best accuracy/speed on CPU). |
| # Speakers | Optional pin for diarization. Leave blank for auto-detection. |
| Language | ISO code (`en`, `es`, …) or blank for auto. Explicit code avoids misdetection on short clips. |
| Voice match threshold | Cosine-similarity cutoff for cross-file identity. Higher = stricter. |
| Output folder | Where TXT transcripts and the batch summary are written. |

---

## Performance Notes

Measured on a Ryzen 9 3900X (12c/24t), no GPU, `int8_float32`, `large-v3-turbo`:

- Real-time factor ≈ **0.25× – 0.35×** on clean speech (i.e. 1 hour of audio → ~15–20 min wall time)
- Real-time factor ≈ **0.45× – 0.6×** on noisy phone/AMR after the denoise/loudnorm preprocess
- Diarization adds roughly **10–15%** on top of transcription time
- Embedding pass is negligible (<5%) since it's one inference per speaker turn, not per frame

A CUDA-capable GPU typically yields a 5–10× end-to-end speedup; `torch.cuda.is_available()` is auto-detected and used if present.

---

## Privacy

All processing runs locally. Audio files, transcripts, and voice embeddings never leave the machine. The only outbound network traffic is the one-time download of model weights from HuggingFace (cached afterward).

---

## Project Structure

```
diarization/
├── app.py              # Pipeline + Gradio UI (single file, ~570 LOC)
├── api_keys.txt        # HF token (gitignored)
├── requirements.txt
├── .gitignore
└── output/             # Generated transcripts
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found` | Install ffmpeg and ensure it's on PATH |
| `401 Unauthorized` from pyannote | Token invalid, or model licences not accepted on HuggingFace |
| Speakers mislabelled across files | Lower the voice-match threshold slightly (e.g. 0.70) |
| Same person split into two identities | Lower the threshold; usually caused by varying mic/noise conditions |
| Very slow on CPU | Switch to `small` or `medium`; or run on a CUDA GPU |

---

## License & Credits

Built on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [PyTorch](https://pytorch.org/), and [Gradio](https://www.gradio.app/). Whisper weights © OpenAI; pyannote weights © pyannote.audio team — see their respective licences.
