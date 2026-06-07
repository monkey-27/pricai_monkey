# Baseline Fairness Audit Agent

Simulated subagent: Baseline Fairness Audit Agent.

Keyword baseline uses overlap over family/cue/action text with a fixed threshold.
TF-IDF baseline uses sklearn when available and bag-of-words cosine otherwise.
PACT methods receive only EpisodeInput, not label or scoring keyword fields.
Leakage scan files: baselines.py, pact.py.

Result: PASS
