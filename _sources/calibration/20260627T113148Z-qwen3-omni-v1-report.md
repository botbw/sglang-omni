# CI Threshold Observation Report

**Calibration commit:** `a5abacd5334c743c81fcd0426628876333bd10f0`
**Branch:** pr883
**Run directory:** `.tune-runs/20260627T113148Z_qwen3-omni-v1_r5`
**Calibration started:** 2026-06-27T11:51:30Z
**Report generated:** 2026-06-27T17:38:05Z

## 1. MMMU Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 64.00 |
| 2 | 50 | 50 | 66.00 |
| 3 | 50 | 50 | 64.00 |
| 4 | 50 | 50 | 66.00 |
| 5 | 50 | 50 | 66.00 |
| **Worst-of-5** | — | — | **64.00** |

## 2. MMMU Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.762 | 83.6 | 7.641 |
| 2 | 50 | 50 | 1.862 | 83.4 | 6.978 |
| 3 | 50 | 50 | 1.789 | 84.9 | 7.451 |
| 4 | 50 | 50 | 1.854 | 82.5 | 7.136 |
| 5 | 50 | 50 | 1.759 | 82.8 | 7.660 |
| **Worst-of-5** | — | — | **1.759** | **82.5** | **7.660** |

## 3. MMMU TALKER Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 80.00 |
| 2 | 20 | 20 | 70.00 |
| 3 | 20 | 20 | 75.00 |
| 4 | 20 | 20 | 75.00 |
| 5 | 20 | 20 | 75.00 |
| **Worst-of-5** | — | — | **70.00** |

## 4. MMMU TALKER Wer

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 18.58 | 8 |
| 2 | 20 | 20 | 23.87 | 5 |
| 3 | 20 | 20 | 22.24 | 6 |
| 4 | 20 | 20 | 23.68 | 5 |
| 5 | 20 | 20 | 21.23 | 6 |
| **Worst-of-5** | — | — | **23.87** | **8** |

## 5. MMMU TALKER Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.609 | 6.4 | 21.573 | 0.5456 |
| 2 | 20 | 20 | 0.662 | 7.3 | 18.617 | 0.4620 |
| 3 | 20 | 20 | 0.650 | 7.6 | 18.524 | 0.4713 |
| 4 | 20 | 20 | 0.666 | 7.5 | 18.749 | 0.4841 |
| 5 | 20 | 20 | 0.692 | 7.4 | 18.818 | 0.4807 |
| **Worst-of-5** | — | — | **0.609** | **6.4** | **21.573** | **0.5456** |

## 6. MMSU Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 2000 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 2000 | 2000 | 70.10 |
| 2 | 2000 | 2000 | 70.70 |
| 3 | 2000 | 2000 | 71.10 |
| 4 | 2000 | 2000 | 70.00 |
| 5 | 2000 | 2000 | 69.90 |
| **Worst-of-5** | — | — | **69.90** |

## 7. MMSU Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 2000 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 2000 | 2000 | 73.431 | 9.5 | 0.217 |
| 2 | 2000 | 2000 | 70.421 | 9.1 | 0.226 |
| 3 | 2000 | 2000 | 70.104 | 9.1 | 0.228 |
| 4 | 2000 | 2000 | 72.738 | 9.4 | 0.219 |
| 5 | 2000 | 2000 | 68.300 | 8.8 | 0.234 |
| **Worst-of-5** | — | — | **68.300** | **8.8** | **0.234** |

## 8. MMSU TALKER Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 40 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 40 | 40 | 62.50 |
| 2 | 40 | 40 | 65.00 |
| 3 | 40 | 40 | 67.50 |
| 4 | 40 | 40 | 62.50 |
| 5 | 40 | 40 | 65.00 |
| **Worst-of-5** | — | — | **62.50** |

## 9. MMSU TALKER Wer

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 40 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 40 | 40 | 1.65 | 0 |
| 2 | 40 | 40 | 2.02 | 0 |
| 3 | 40 | 40 | 1.42 | 1 |
| 4 | 40 | 40 | 2.38 | 0 |
| 5 | 40 | 40 | 2.57 | 0 |
| **Worst-of-5** | — | — | **2.57** | **1** |

## 10. MMSU TALKER Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 40 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 40 | 40 | 1.377 | 5.8 | 10.440 | 0.5774 |
| 2 | 40 | 40 | 1.316 | 5.3 | 11.399 | 0.6275 |
| 3 | 40 | 40 | 1.252 | 5.2 | 12.072 | 0.6445 |
| 4 | 40 | 40 | 1.323 | 5.3 | 11.412 | 0.6323 |
| 5 | 40 | 40 | 1.277 | 5.2 | 11.775 | 0.6412 |
| **Worst-of-5** | — | — | **1.252** | **5.2** | **12.072** | **0.6445** |

## 11. TTS Wer

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.24 | 0 |
| 2 | 50 | 50 | 3.37 | 0 |
| 3 | 50 | 50 | 2.13 | 0 |
| 4 | 50 | 50 | 1.77 | 0 |
| 5 | 50 | 50 | 2.13 | 0 |
| **Worst-of-5** | — | — | **3.37** | **0** |

## 12. TTS Utmos

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | UTMOS mean |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 4.2487 |
| 2 | 50 | 50 | 4.2817 |
| 3 | 50 | 50 | 4.2752 |
| 4 | 50 | 50 | 4.2388 |
| 5 | 50 | 50 | 4.2584 |
| **Worst-of-5** | — | — | **4.2388** |

## 13. TTS Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 5.548 | 5.6 | 2.631 | 0.8716 |
| 2 | 50 | 50 | 5.520 | 5.5 | 2.655 | 0.8633 |
| 3 | 50 | 50 | 5.567 | 5.6 | 2.635 | 0.8860 |
| 4 | 50 | 50 | 5.641 | 5.5 | 2.644 | 0.9077 |
| 5 | 50 | 50 | 5.317 | 5.2 | 2.803 | 0.9536 |
| **Worst-of-5** | — | — | **5.317** | **5.2** | **2.803** | **0.9536** |

## 14. VIDEOAMME Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 68.00 |
| 2 | 50 | 50 | 68.00 |
| 3 | 50 | 50 | 66.00 |
| 4 | 50 | 50 | 68.00 |
| 5 | 50 | 50 | 68.00 |
| **Worst-of-5** | — | — | **66.00** |

## 15. VIDEOAMME Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.797 | 6.7 | 7.734 |
| 2 | 50 | 50 | 1.904 | 6.9 | 7.322 |
| 3 | 50 | 50 | 1.825 | 6.5 | 7.658 |
| 4 | 50 | 50 | 1.804 | 6.8 | 7.665 |
| 5 | 50 | 50 | 1.689 | 6.2 | 8.234 |
| **Worst-of-5** | — | — | **1.689** | **6.2** | **8.234** |

## 16. VIDEOAMME TALKER TP2 Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 10 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 50.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 17. VIDEOAMME TALKER TP2 Wer

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 10 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 5.48 | 0 |
| 2 | 10 | 10 | 0.80 | 0 |
| 3 | 10 | 10 | 1.52 | 0 |
| 4 | 10 | 10 | 2.89 | 0 |
| 5 | 10 | 10 | 1.01 | 0 |
| **Worst-of-5** | — | — | **5.48** | **0** |

## 18. VIDEOAMME TALKER TP2 Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 10 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.058 | 0.3 | 168.235 | 12.8415 |
| 2 | 10 | 10 | 0.062 | 0.3 | 159.234 | 12.5897 |
| 3 | 10 | 10 | 0.058 | 0.3 | 170.360 | 12.4779 |
| 4 | 10 | 10 | 0.058 | 0.3 | 166.749 | 11.9758 |
| 5 | 10 | 10 | 0.068 | 0.3 | 145.754 | 10.8548 |
| **Worst-of-5** | — | — | **0.058** | **0.3** | **170.360** | **12.8415** |

## 19. VIDEOMME Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 58.00 |
| 2 | 50 | 50 | 60.00 |
| 3 | 50 | 50 | 58.00 |
| 4 | 50 | 50 | 58.00 |
| 5 | 50 | 50 | 58.00 |
| **Worst-of-5** | — | — | **58.00** |

## 20. VIDEOMME Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 50 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.134 | 9.2 | 12.161 |
| 2 | 50 | 50 | 1.205 | 9.7 | 11.465 |
| 3 | 50 | 50 | 1.087 | 8.6 | 12.684 |
| 4 | 50 | 50 | 1.117 | 9.1 | 12.306 |
| 5 | 50 | 50 | 1.160 | 9.3 | 11.908 |
| **Worst-of-5** | — | — | **1.087** | **8.6** | **12.684** |

## 21. VIDEOMME TALKER Accuracy

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 60.00 |
| 2 | 20 | 20 | 60.00 |
| 3 | 20 | 20 | 65.00 |
| 4 | 20 | 20 | 60.00 |
| 5 | 20 | 20 | 60.00 |
| **Worst-of-5** | — | — | **60.00** |

## 22. VIDEOMME TALKER Wer

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 3.10 | 0 |
| 2 | 20 | 20 | 4.00 | 0 |
| 3 | 20 | 20 | 2.14 | 1 |
| 4 | 20 | 20 | 1.05 | 0 |
| 5 | 20 | 20 | 1.98 | 0 |
| **Worst-of-5** | — | — | **4.00** | **1** |

## 23. VIDEOMME TALKER Speed

— 2× NVIDIA H100 80GB HBM3 from precheck.json, 20 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.966 | 3.6 | 11.972 | 1.3731 |
| 2 | 20 | 20 | 0.920 | 3.6 | 12.134 | 1.1500 |
| 3 | 20 | 20 | 0.931 | 3.2 | 13.059 | 1.5125 |
| 4 | 20 | 20 | 0.887 | 3.4 | 12.348 | 1.4727 |
| 5 | 20 | 20 | 0.921 | 3.5 | 12.590 | 1.2337 |
| **Worst-of-5** | — | — | **0.887** | **3.2** | **13.059** | **1.5125** |

## Provenance

- Model: qwen3-omni-v1
- Calibration commit: `a5abacd5334c743c81fcd0426628876333bd10f0`
- Branch: pr883 (dirty) — see `workspace.diff`
- Calibration started: 2026-06-27T11:51:30Z
- Venv Python: /github/home/calibration/omni/bin/python (flag)
- sglang 0.5.12.post1 · torch 2.11.0+cu130
- GPU: 2× NVIDIA H100 80GB HBM3
- tune-ci-thresholds v0.6.0
- Report generated: 2026-06-27T17:38:05Z
