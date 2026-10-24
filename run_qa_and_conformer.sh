#!/bin/bash
# IndicConformer on the translation set + the Belebele-Fleurs listening-comprehension eval. One model in memory at a time.
set -e
export PYTHONPATH=src
CONF=ai4bharat/indic-conformer-600m-multilingual
# --- ASR passes ---
python src/run_asr.py --asr $CONF --decoder rnnt --tag asr-conf-rnnt
python src/run_asr.py --asr $CONF --decoder ctc  --tag asr-conf-ctc
python src/qa_asr.py --tag gold
python src/qa_asr.py --tag small --asr openai/whisper-small
python src/qa_asr.py --tag turbo --asr openai/whisper-large-v3-turbo --dtype fp16
python src/qa_asr.py --tag conf  --asr $CONF --decoder rnnt
# --- translation + answering ---
python src/qa_mt.py --tags gold small turbo conf
python src/qa_answer.py --tags gold small turbo conf
# --- IndicConformer variants of the main translation evaluation (RNNT chosen a priori) ---
python src/respond.py --src asr-conf-rnnt --out conf
python src/respond.py --src asr-conf-rnnt --out conf+mt --ground
python src/mt_only.py --src asr-conf-rnnt --out mtonly-conf
