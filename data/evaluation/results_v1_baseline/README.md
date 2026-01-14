# V1: ベースライン評価結果（改善前）

## 評価日時
2026-01-13

## 設定パラメータ

### NaiveRAG
- **top_k**: 5
- **chunk_size**: 500
- **chunk_overlap**: 100
- **max_tokens**: 8192
- **model**: gpt-5-mini

### プロンプト
- 基本的な指示のみ
- 画像・表・数値に関する詳細な指示なし

## 結果

### Overall
- **Accuracy**: 50% (5/10問正解)
- **LLM Judge**: 0.72

### カテゴリ別
| カテゴリ | 正解数 | 精度 |
|----------|--------|------|
| Paragraph | 2/4 | 50% |
| Image | 2/4 | 50% |
| Table | 1/2 | 50% |

### 質問別結果
| Q# | 正解 | タイプ |
|----|------|--------|
| Q0 | ❌ | Paragraph |
| Q1 | ✅ | Image |
| Q2 | ❌ | Image |
| Q3 | ✅ | Paragraph |
| Q4 | ❌ | Paragraph |
| Q5 | ❌ | Paragraph |
| Q6 | ✅ | Image |
| Q7 | ✅ | Image |
| Q8 | ❌ | Table |
| Q9 | ✅ | Table |

## ファイル
- `evaluation_summary_NaiveRAG.json` - サマリー
- `evaluation_detail_NaiveRAG.csv` - 詳細結果

## 次のステップ
改善後の結果は `../results_v2_improved/` を参照
