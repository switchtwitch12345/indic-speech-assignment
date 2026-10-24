#!/bin/bash
# Resume, in priority order (most important results first; noisy-audio LLM stages last and optional).
set -e
export PYTHONPATH=src
python src/qa_answer.py --tags turbo conf                                              # finish the listening-comprehension eval
python src/respond.py --src asr-conf-rnnt --out conf+mt --ground --conds clean         # best expected system, clean
python src/mt_only.py --src asr-conf-rnnt --out mtonly-conf --conds clean              # no-LLM control, clean
python src/respond.py --src asr-conf-rnnt --out conf --conds clean                     # plain LLM, clean
python src/mt_only.py --src asr-conf-rnnt --out mtonly-conf --conds noisy              # --- noisy, optional ---
python src/respond.py --src asr-conf-rnnt --out conf+mt --ground --conds noisy
python src/respond.py --src asr-conf-rnnt --out conf --conds noisy
