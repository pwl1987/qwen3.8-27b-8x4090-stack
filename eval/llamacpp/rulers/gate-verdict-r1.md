| 轴 | 指标 | 基线 | 门禁 | Δ | 判定 |
|---|---|---|---|---|---|
| humaneval | pass_at_1 | 0.8476 | 0.9573 | +11.0pp | PASS |
| xfc | accuracy | 0.6350 | 0.7350 | +10.0pp | PASS |
| gsm8k | accuracy | 0.0000 | 0.9400 | +94.0pp | PASS |
| ifeval | accuracy | 0.0000 | 0.7600 | +76.0pp | PASS |
| needle | needle_64k.hit | 1 | 0 | 0/1 | FAIL |
| needle | needle_195k.hit | 0 | None | MISSING | — |
| longgen | approx_tps | 43.5000 | 33.3000 | -23.4% | FAIL |

结论: FAIL
  - needle.needle_64k.hit 回退 0/1
  - needle.needle_195k.hit 数据缺失
  - longgen.approx_tps 回退 -23.4%
