#!/bin/bash
# Reproduces all improvement/ablation runs. Each stage keeps a single model in memory.
set -e
export PYTHONPATH=src
python src/respond.py --src base --out base+mt --ground                  # B: NLLB-grounded LLM, same whisper-small hypotheses
python src/respond.py --src gold --out oracle+mt --ground --conds oracle # B on gold transcripts (isolates LLM comprehension)
python src/run_asr.py --asr openai/whisper-large-v3-turbo --tag asr-turbo --dtype fp16   # A: ASR upgrade
python src/respond.py --src asr-turbo --out turbo                        # A
python src/respond.py --src asr-turbo --out turbo+mt --ground            # A+B
# Control: MT only (no LLM), on the same transcripts
python src/mt_only.py --src base --out mtonly-base
python src/mt_only.py --src asr-turbo --out mtonly-turbo
python src/mt_only.py --src gold --out mtonly-oracle --conds oracle
