# cacao_healty_check

比較実験用のカカオ病害画像分類プロジェクトです。

同じデータ分割・学習条件で、次の3モデルを比較します。

1. `ResNet50`
2. `ResNet50 + CBAM`
3. `ResNet50 + LGF-CBAM`

対象はまず `healthy / black_pod / other` の3クラスを想定しています。`ImageFolder`形式ならクラスを増やして多クラス分類にも使えます。

## データ配置

学習用データはGitHubにアップロードせず、次の形式でローカルに配置してください。

```text
data/cacao/
├── train/
│   ├── black_pod/
│   ├── healthy/
│   └── other/
├── val/
│   ├── black_pod/
│   ├── healthy/
│   └── other/
└── test/
    ├── black_pod/
    ├── healthy/
    └── other/
```

クラス名はフォルダ名から自動取得されます。データセットに含まれる画像・重み・結果は `.gitignore` で除外しています。

## セットアップ

Python 3.11〜3.13を推奨します。Windows CPUのみで動かす場合も同じ手順です。

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3モデルを比較

```bash
python -m src.train \
  --data-root data/cacao \
  --output-dir results/compare \
  --models resnet50 cbam lgf_cbam \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 0
```

CPUで最初の動作確認をする場合は、次のように短く回します。

```bash
python -m src.train \
  --data-root data/cacao \
  --output-dir results/smoke \
  --models resnet50 lgf_cbam \
  --epochs 1 \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --batch-size 4 \
  --no-pretrained
```

## 出力

各モデルについて以下を保存します。

- `best.pt`: validation macro-F1が最良の重み
- `metrics.json`: Accuracy、Macro-F1、BPR recall、クラス別指標、CPU推論時間
- `history.json`: epochごとのlossと評価値

比較ではAccuracyだけでなく、特に `black_pod` のRecallとMacro-F1を重視してください。画像単位のランダム分割だけでは精度が過大評価される可能性があるため、実運用前には圃場・木・撮影日単位の外部テストを追加してください。

## モデルの考え方

- CBAMはchannel attention（何を見るか）とspatial attention（どこを見るか）を順番に適用します。
- LGF-CBAMは両方の特徴を作り、入力特徴に応じて学習可能なゲートで融合します。
- attentionはResNet50の`layer4`後、global average pooling前に追加しています。

## 参考

- [LGF-CBAM paper](https://doi.org/10.1371/journal.pone.0348147)
- [Cocoa-Pod-Disease-Detection OSS](https://github.com/Unclefee/Cocoa-Pod-Disease-Detection)
- [Cocoa image dataset / OSF](https://osf.io/2fw6g)

