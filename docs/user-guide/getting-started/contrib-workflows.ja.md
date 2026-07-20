# コミュニティ提供ワークフロー

Qubex の追加ワークフローの一部は、`Experiment` のコアメソッドではなく、`qubex.contrib` 配下のコミュニティ提供関数として提供されています。

このページは主に既存ユーザー向けの移行メモです。古い notebook や script で `Experiment` の helper が見つからなくなった場合は、対応する contrib 関数を使い、最初の引数に `exp` を渡してください。

```python
from qubex import contrib
```

## 移動した API

古い notebook や script を更新するときは、次の対応表を使ってください。

| `exp` での旧呼び出し | 新しい contrib 関数 |
| --- | --- |
| `exp.measure_cr_crosstalk(...)` | `contrib.measure_cr_crosstalk(exp, ...)` |
| `exp.cr_crosstalk_hamiltonian_tomography(...)` | `contrib.cr_crosstalk_hamiltonian_tomography(exp, ...)` |
| `exp._simultaneous_measurement_coherence(...)` | `contrib.simultaneous_coherence_measurement(exp, ...)` |
| `exp._stark_t1_experiment(...)` | `contrib.stark_t1_experiment(exp, ...)` |
| `exp._stark_ramsey_experiment(...)` | `contrib.stark_ramsey_experiment(exp, ...)` |
| `exp.purity_benchmarking(...)` | `contrib.purity_benchmarking(exp, ...)` |
| `exp.interleaved_purity_benchmarking(...)` | `contrib.interleaved_purity_benchmarking(exp, ...)` |

## JPA 校正

### 基本的な呼び出し

公開 API に必要なのは、experiment と qubit ラベル 1 個だけです。

```python
from qubex import contrib

result = contrib.calibrate_jpa(exp, "Q22")
print(result["optimal_parameters"])
```

`"Q22"` は readout MUX を特定するための anchor です。校正対象が Q22 だけになるわけではありません。同じ MUX に属する active かつ valid な全 qubit を測定し、その全 peer が制約を満たす点だけを候補にします。デフォルトの探索範囲は MUX の設定値から生成され、coarse scan の後、必要なら局所的な fine scan を実行します。

### 完全 OFF baseline と選択条件

候補点の scan を始める前に、DC voltage `0.0` V、pump amplitude `0.0` の完全 OFF baseline を測定します。amplitude が 0 の間は pump frequency は結果に影響しません。baseline と候補点では、対象 qubit、readout amplitude の選択、readout duration、shot 数、shot interval をすべて同じにするため、両者を直接比較できます。

各 peer qubit・各 grid point について、次の 2 つの raw 値を記録します。

- `score`: g/e 状態の IQ cloud 間の距離を noise で規格化した値。大きいほど良い値です。
- `flatness`: g/e それぞれの IQ cloud の主軸方向の標準偏差比のうち大きい方。`1.0` は円形で、大きいほど細長い cloud です。

さらに peer ごとに、OFF baseline に対する相対値を計算します。

```text
score_gain      = 候補点の score / 完全 OFF の score
flatness_ratio  = 候補点の flatness / 完全 OFF の flatness
```

デフォルトでは、全 peer について `score_gain >= 1.05` かつ `flatness_ratio <= 1.1` となる点だけが valid です。その valid point の中から、worst peer の score gain が最大になる点を選びます。つまり、同じ MUX 上のある qubit の大きな改善によって、別の qubit の悪化が隠されることはありません。

測定 notebook では、判定条件を明示して呼び出せます。

```python
result = contrib.calibrate_jpa(
    exp,
    "Q22",
    n_shots=512,
    minimum_score_gain=1.05,
    maximum_flatness_ratio=1.1,
    flatness_threshold=None,
)
```

- `minimum_score_gain` は、全 peer に要求する「候補点 / OFF」の score 比の下限です。たとえば `1.05` なら、各 qubit に 5% 以上の改善を要求します。
- `maximum_flatness_ratio` は、各 peer の flatness がその qubit 自身の OFF baseline からどこまで増えてよいかを制限します。たとえば `1.1` なら増加を 10% まで許します。
- `flatness_threshold` は相対条件に追加できる、任意の絶対 flatness 上限です。デフォルトは `None` で、絶対上限を無効にします。独立した根拠のある絶対上限が必要な場合だけ、たとえば `1.5` を指定してください。OFF cloud が元から異方的な系では、`1.2` のような固定値を使うと、JPA が異方性の原因ではないのに全点を棄却することがあります。

`readout_amplitude` を省略すると、各 peer はそれぞれの設定値を使います。1 個の値を渡すと、評価する全 peer の amplitude を同じ値で上書きします。ON/OFF は必ず同じ readout 条件で比較する必要がありますが、自動測定される baseline はこの条件を保証します。

### Result の確認

返される `Result` には、選択された点だけでなく、その判断に使ったデータも含まれます。

```python
print(result["baseline"])
print(result["score"], result["score_gain"])
print(result["scores_by_qubit"])
print(result["score_gains_by_qubit"])
print(result["flatness_by_qubit"])
print(result["flatness_ratios_by_qubit"])

scan = result["fine_scan"] or result["coarse_scan"]
raw_q22 = scan["scores"]["Q22"]
gain_q22 = scan["score_gains"]["Q22"]
flatness_q22 = scan["flatness"]["Q22"]
flatness_ratio_q22 = scan["flatness_ratios"]["Q22"]
valid = scan["valid_mask"]
```

raw scan 配列には物理的な score と flatness が保持されます。relative 配列には 1 回の完全 OFF baseline に対する比が保持され、`aggregate_score_gain` は各 grid point の worst-peer gain を表します。`valid_mask` では、有効になっている制約をすべて満たした点を確認できます。

### 条件を満たす点がない場合

scan が完了しても、許容できる JPA 設定が存在するとは限りません。全 peer の条件を満たす点が 1 個もない場合、`calibrate_jpa` は `JPAConstraintError` を送出します。完了済みの raw/relative scan は破棄されず、`diagnostics` から確認できます。

```python
from qubex import contrib

try:
    result = contrib.calibrate_jpa(
        exp,
        "Q22",
        n_shots=512,
        minimum_score_gain=1.05,
        maximum_flatness_ratio=1.1,
    )
except contrib.JPAConstraintError as exc:
    diagnostics = exc.diagnostics
    print(diagnostics["failure_stage"])
    print(diagnostics["baseline"])

    coarse = diagnostics["coarse_scan"]
    print(coarse["aggregate_score_gain"])
    print(coarse["valid_mask"])
```

この診断情報から、有効な領域が探索範囲の外にあったのか、特定の peer が悪化したのか、意図した制約が厳しすぎたのかを判断してください。数値が最大というだけで invalid point を校正値として採用してはいけません。

この workflow は意図的に測定専用です。成功時も候補値を返すだけで、選択した DC/pump 設定を適用したままにはせず、`jpa_params.yaml` を含む parameter file にも書き込みません。成功・失敗のどちらでも、DC context は以前の電圧を復元し、操作した output を disable にします。返された候補を確認・検証してから、実験室の通常の configuration 手順で適用・保存してください。

## Simultaneous coherence

```python
import numpy as np
from qubex import contrib

results = contrib.simultaneous_coherence_measurement(
    exp,
    targets=[Q0, Q1],
    time_range=np.arange(0, 20_001, 1000),
    n_shots=1024,
)

t1_result = results["T1"]
t1_result.plot()
```

## Stark-driven characterization

```python
from qubex import contrib

stark_result = contrib.stark_t1_experiment(
    exp,
    targets=[Q0],
    stark_detuning=0.05,
    stark_amplitude=0.1,
    n_shots=1024,
)

stark_result.plot()
```

## Purity benchmarking

```python
from qubex import contrib

pb_result = contrib.purity_benchmarking(
    exp,
    targets=[Q0],
    n_shots=1024,
)

print(pb_result)
```
