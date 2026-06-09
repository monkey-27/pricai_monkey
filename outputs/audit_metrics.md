# Metric Audit Agent

Simulated subagent: Metric Audit Agent.
False-trigger metrics are split into near-miss, wrong-scope, wrong-contract, excluding-contract-swap, and including-contract-swap rates.
Contract-swap false-trigger rate is reported separately because wrong-contract activations were hidden by the old aggregate.
Strict end-to-end success requires explicit conflict detection; behavioral end-to-end success allows safe conflict behavior without explicit detection.
Conflict detection and conflict-safe action are separate metrics.
Wrong-contract action completion and irrelevant action completion are reported separately from target action completion.
Zero division uses deterministic 0.0 fallback.
Weighted utility implements the preregistered signs and penalties.
Result: PASS
