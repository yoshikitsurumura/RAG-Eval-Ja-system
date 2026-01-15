# RAGシステム精度改善実装レポート

## 実装日時
2026-01-14

## 背景
初回評価（NaiveRAG）で50%の精度（10問中5問正解）を記録。以下の問題点が確認された:
- 画像・表からのデータ抽出失敗（Q2, Q8）
- トレンド方向の誤解釈（Q5）
- 回答の詳細度のミスマッチ（Q4）
- 具体的数値の欠落（Q0）

## 制約条件
- **LLMモデルは変更不可**: gpt-5-miniを継続使用（課題要件）
- モデル以外の全ての改善は許可

## 実装した改善策

### 1. 検索件数（top_k）の増加

#### 目的
より多くのコンテキスト情報を提供することで、必要な情報が取得される確率を向上させる。

#### 変更内容

**NaiveRAG (`src/rag/naive_rag.py`)**
```python
# 変更前
top_k: int = 5

# 変更後
top_k: int = 10
```

**AgenticRAG (`src/rag/agentic_rag.py`)**
```python
# 変更前
top_k: int = 3

# 変更後
top_k: int = 7
```

#### 期待効果
- 情報カバレッジの向上（特に複数箇所に情報が散在している場合）
- 表・画像データを含むチャンクが取得される確率の向上
- NaiveRAG: 2倍、AgenticRAG: 2.3倍のコンテキスト

### 2. プロンプトの強化

#### 目的
LLMに対して、画像・表の扱い方、回答の詳細度、数値の正確性について明確な指示を与える。

#### 変更内容

**Naive RAGシステムプロンプト (`src/generation/prompt_templates.py`)**

追加した重要な注意事項:
```python
【重要な注意事項】
- 画像やグラフの場合は、具体的な数値を必ず確認して言及すること
- 表の場合は、行と列を正確に読み取ること
- トレンド（増加・減少）を述べる際は、数値の変化を慎重に確認すること
- 質問が「簡潔に」「要約して」と求めている場合は3-4文以内で端的に回答すること
- 質問が「具体的に」「詳しく」と求めている場合は具体例と数値を含めて詳細に説明すること
- 数値データは正確に引用し、単位も含めること
```

**Agentic RAG回答生成プロンプト (`src/generation/prompt_templates.py`)**

同様の注意事項を`ANSWER_GENERATOR_PROMPT`にも追加。

#### 期待効果
- 画像・表からの数値抽出精度の向上
- トレンド解釈の誤りの減少
- 質問の要求レベルに応じた適切な回答長
- 数値の単位付与による正確性の向上

### 3. チャンクサイズとオーバーラップの調整

#### 目的
官公庁文書は文脈が長い傾向があるため、チャンクサイズを大きくして情報の分断を防ぐ。
オーバーラップを増やして境界での情報欠落を防止。

#### 変更内容

**RecursiveTextSplitter (`src/ingestion/text_splitter.py`)**
```python
# 変更前
chunk_size: int = 500
chunk_overlap: int = 100

# 変更後
chunk_size: int = 800
chunk_overlap: int = 200
```

**TableAwareTextSplitter (`src/ingestion/text_splitter.py`)**
同様の変更を適用。

**get_text_splitter関数のデフォルト値**
同様の変更を適用。

#### 期待効果
- 長い文脈を保持したまま分割（60%増）
- 境界での情報損失の削減（オーバーラップ2倍）
- 表と周辺テキストの関連性保持の向上

### 4. AgenticRAGの反復回数増加

#### 目的
品質評価に基づく改善の機会を増やす。

#### 変更内容

**AgenticRAG (`src/rag/agentic_rag.py`)**
```python
# 変更前
max_iterations: int = 1  # 改善なし

# 変更後
max_iterations: int = 2  # 1回の改善機会
```

#### 期待効果
- 初回回答が不十分な場合に改善のチャンス
- 自己評価（Self-Reflection）機能の活用
- 品質閾値（4/5）未満の回答の改善

## 改善の根拠

### 初回評価の失敗パターン分析

| 質問ID | タイプ | 失敗理由 | 対応する改善策 |
|--------|--------|----------|----------------|
| Q0 | パラグラフ | 具体的数値の欠落 | プロンプト強化、top_k増加 |
| Q2 | 画像 | 画像データ抽出失敗 | プロンプト強化、top_k増加 |
| Q4 | パラグラフ | 詳細すぎる回答 | プロンプト強化（要約指示） |
| Q5 | パラグラフ | トレンド方向逆 | プロンプト強化（数値確認） |
| Q8 | 表 | 表データ抽出失敗 | プロンプト強化、top_k増加 |

### 他の検討した改善案（今回は未実装）

以下は将来的な改善案として検討可能:

**Hybrid Search（ベクトル + キーワード）**
- メリット: 固有名詞・数値の正確なマッチ
- デメリット: Qdrantでのハイブリッド実装が複雑
- 優先度: 中

**Reranking**
- メリット: 検索精度の向上
- デメリット: 追加のLLM呼び出しコスト
- 優先度: 中

**gpt-4oへの変更**
- メリット: マルチモーダル（画像直接入力）、高推論能力
- デメリット: モデル変更禁止（課題要件）
- 優先度: 不可

**セマンティックチャンキング**
- メリット: 意味的なまとまりで分割
- デメリット: 実装コストと複雑性
- 優先度: 低

## 実装後の検証計画

### 1. データ再インジェスト
チャンクサイズ変更により、既存のベクトルDBを更新する必要がある。

```bash
# ベクトルDBのコレクション削除
docker compose exec api python scripts/setup_qdrant.py --reset

# データ再インジェスト
docker compose exec api python scripts/ingest_documents.py
```

### 2. 評価実行

**NaiveRAG評価**
```bash
docker compose exec api python scripts/run_evaluation.py --rag-types naive --limit 10
```

**AgenticRAG評価**
```bash
docker compose exec api python scripts/run_evaluation.py --rag-types agentic --limit 10
```

### 3. 期待される改善

**目標精度**
- NaiveRAG: 50% → 60-70%（+20%ポイント）
- AgenticRAG: 現在評価中 → 70-80%を期待

**特に改善を期待する質問**
- Q2（画像）: プロンプト強化により数値抽出向上
- Q4（詳細度）: 「簡潔に」の指示に従った短い回答
- Q5（トレンド）: 数値確認の指示により方向性正確化
- Q8（表）: top_k増加により表データ取得確率向上

## まとめ

### 実装内容
1. ✅ top_k増加（NaiveRAG: 5→10, AgenticRAG: 3→7）
2. ✅ プロンプト強化（画像・表・数値・詳細度の指示追加）
3. ✅ チャンクサイズ調整（500→800, オーバーラップ100→200）
4. ✅ AgenticRAG反復回数増加（1→2）

### 影響範囲
- **変更ファイル**: 3ファイル
  - `src/rag/naive_rag.py`
  - `src/rag/agentic_rag.py`
  - `src/generation/prompt_templates.py`
  - `src/ingestion/text_splitter.py`
- **破壊的変更**: チャンクサイズ変更のため、ベクトルDB再構築が必要
- **モデル変更**: なし（gpt-5-mini継続）

### 次のステップ
1. データ再インジェスト実行
2. 評価実行（NaiveRAG, AgenticRAG）
3. 結果分析と追加改善の検討