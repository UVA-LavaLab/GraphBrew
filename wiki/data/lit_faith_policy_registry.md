# cache-policy vocabulary registry — gate 255

Status: `active`

Totals: canonical=8  harvested_POLICIES=43  harvested_ALL=2  violations=0

Canonical four-tuple: `('LRU', 'SRRIP', 'GRASP', 'POPT')`
Anchor triplet: `('GRASP', 'LRU', 'SRRIP')`

## Canonical tokens

- `ECG` — family=`graph_aware` paper_label=`ECG` aliases=['Ecg', 'ecg']
- `FIFO` — family=`baseline` paper_label=`FIFO` aliases=['Fifo', 'fifo']
- `GRASP` — family=`graph_aware` paper_label=`GRASP` aliases=['Grasp', 'grasp', 'G.R.A.S.P']
- `LFU` — family=`baseline` paper_label=`LFU` aliases=['Lfu', 'lfu']
- `LRU` — family=`baseline` paper_label=`LRU` aliases=['Lru', 'lru', 'L.R.U', 'LRU_cache']
- `POPT` — family=`graph_aware` paper_label=`P-OPT` aliases=['Popt', 'popt', 'P_OPT']
- `RANDOM` — family=`baseline` paper_label=`Random` aliases=['Random', 'RAND', 'RND']
- `SRRIP` — family=`baseline` paper_label=`SRRIP` aliases=['Srrip', 'srrip', 'SRIP', 'S-RRIP']

## ECG arms

- `ECG:DBG_ONLY` — parent=`ECG` — Debug-only ECG arm — emits diagnostic counters; not a production policy.
- `ECG:DBG_PRIMARY` — parent=`ECG` — Debug primary ECG arm (uncharged baseline).
- `ECG:DBG_PRIMARY_CHARGED` — parent=`ECG` — Debug primary with charged-overhead accounting.
- `ECG:ECG_COMBINED` — parent=`ECG` — ECG combining multiple sub-scorers into one decision.
- `ECG:ECG_EMBEDDED` — parent=`ECG` — ECG embedded inside the L3 substrate directly.
- `ECG:ECG_EPOCH_EMBEDDED` — parent=`ECG` — ECG embedded with epoch-bounded retraining.
- `ECG:POPT_PRIMARY` — parent=`ECG` — ECG with POPT as the primary scorer (gate-239 parity arm).
- `ECG:POPT_TIE` — parent=`ECG` — ECG with POPT tie-breaking on equal scores.
- `POPT_CHARGED` — parent=`POPT` — POPT with the charged-overhead accounting model active (ROI substrate ablation arm).
