# Pre-check flag-rate audit

Counts only -- see reflex_sentry.models.precheck.audit. Higher is better on dangerous/generalization rows, lower is better on benign rows.

| Slice | Flag rate (k/n) |
|---|---|
| train_benign | 0.001 (3/3314) |
| tuning_ood_benign (wildguardmix+or_bench, used for tuning) | 0.002 (3/2000) |
| toxic_chat_ordinary_benign (report only, never tuned) | 0.008 (16/2000) |
| toxic_chat_ordinary_benign:de | 0.000 (0/7) |
| toxic_chat_ordinary_benign:en_or_other | 0.005 (10/1955) |
| toxic_chat_ordinary_benign:es_pt | 0.182 (6/33) |
| toxic_chat_ordinary_benign:fr | 0.000 (0/4) |
| toxic_chat_ordinary_benign:unknown | 0.000 (0/1) |
| val_benign_hard_negative | 0.000 (0/198) |
| val_benign_other | n/a (n=0) |
| val_dangerous | 0.000 (0/48) |
| test_benign_hard_negative | 0.000 (0/208) |
| test_benign_other | 0.000 (0/1) |
| test_dangerous | 0.000 (0/47) |
| test_ood_benign_hard_negative | 0.000 (0/19) |
| test_ood_benign_other | n/a (n=0) |
| test_ood_dangerous | 0.000 (0/121) |
| test_evasion:authority | 0.000 (0/77) |
| test_evasion:base64 | 1.000 (77/77) |
| test_evasion:fiction | 0.000 (0/77) |
| test_evasion:leetspeak | 1.000 (77/77) |
| test_evasion:prefix_noise | 0.000 (0/77) |
| test_evasion:research | 0.000 (0/77) |
| test_evasion:roleplay | 0.000 (0/77) |
| test_evasion:split | 0.000 (0/77) |
| generalization:hex | 1.000 (246/246) |
| generalization:rot13 | 1.000 (246/246) |
| generalization:url_percent | 1.000 (246/246) |
| generalization:base32 | 1.000 (246/246) |
| generalization:char_spacing | 1.000 (246/246) |
| generalization:reversed | 1.000 (246/246) |
| generalization:homoglyph | 1.000 (246/246) |
