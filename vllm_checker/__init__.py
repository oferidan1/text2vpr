"""Tools for re-checking SAM3 missing objects with an LLM.

The main entrypoint is `vllm_checker.main` which reads a
`sam3_realtime_progress*.csv`, asks an LLM per-missing-object
"yes/no" questions, and writes an augmented CSV with an extra
column listing objects the LLM still considers missing.

The LLM backend is implemented in `llm_client.py` and is intentionally
simple and modular so you can swap in a different model (e.g. a vLLM
server hosting Phi-4-vision) without touching the CSV plumbing code.
"""

