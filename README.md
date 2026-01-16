# Laboro RAG System

株式会社Laboro.AI LLMエンジニア コーディング課題 - RAGシステムの実装

## 概要

日本語RAG評価データセット（RAG-Evaluation-Dataset-JA）を知識源とした実践的なRAGシステムです。

### 実装内容

- **Naive RAG**: 単純なベクトル検索と回答生成を行うベースライン実装
- **Agentic RAG**: LLMエージェントが自律的に検索戦略を制御する発展版実装

## 実装バージョンと評価結果

| Version | 実装内容 | 主要技術 | 精度 | ステータス |
|---------|---------|---------|------|-----------|
| **V1** | Baseline (Naive RAG) | ベクトル検索のみ | 50.0% | ✅ 完了 |
| **V2** | パラメータ改善 | top_k拡大、chunk調整、プロンプト強化 | 60.0% | ✅ 完了 |
| **V3** | Hybrid Search + Reranking | BM25 + Vector + RRF、日本語Reranker | 60.0% | ✅ 完了 |
| **V4** | Vision + Auto-Tagging | GPT-4o画像解析、自動分類、Commander AI | - | ⚠️ 実装済み・未検証 |

### 各バージョンの詳細

#### V1: Baseline (50%)
- ベクトル検索のみのシンプルな実装
- 詳細: [data/evaluation/results_v1_baseline/](data/evaluation/results_v1_baseline/)

#### V2: パラメータ改善 (60%)
- **改善点**:
  - top_k: 5 → 10
  - chunk_size: 500 → 800
  - プロンプト強化（数値・グラフの読み取り指示）
- **効果**: Paragraph精度が50% → 75%に向上
- 詳細: [data/evaluation/results_v2_improved/](data/evaluation/results_v2_improved/)

#### V3: Hybrid Search + Reranking (60%)
- **改善点**:
  - ハイブリッド検索（BM25 + ベクトル検索 + RRF）
  - 日本語Reranker (`cl-nagoya/ruri-reranker-small`)
  - Alpha: 0.7（ベクトル70% : BM25 30%）
- **効果**: Paragraph精度が75%を維持、全体60%
- **課題**: Image精度が75% → 50%に低下
- 詳細: [data/evaluation/results_v3_hybrid_reranking/](data/evaluation/results_v3_hybrid_reranking/)

#### V4: Vision + Auto-Tagging（実装済み・未検証）
- **実装内容**:
  - Vision API（GPT-4o）による画像解析・キャプション生成
  - ドキュメント自動分類（金融、製造、食品、IT、行政）
  - Commander AIによる動的な検索フィルタリング
- **ステータス**: コード実装完了、APIキー無効化のため評価未実施
- **期待効果**: Image精度の回復（50% → 75%以上）
- 詳細: [docs/PROJECT_WRAP_UP_REPORT.md](docs/PROJECT_WRAP_UP_REPORT.md)

### カテゴリ別精度推移

| カテゴリ | V1 | V2 | V3 | 備考 |
|---------|----|----|----|----|
| **Overall** | 50% | 60% | 60% | V2で+10%向上 |
| Paragraph | 50% | 75% | 75% | V2/V3で大幅改善 |
| Image | 50% | 50% | 50% | V3で一時低下も調整済み |
| Table | 50% | 50% | 50% | 改善余地あり |

## クイックスタート

### 1. 環境変数設定

```bash
cp .env.example .env
# .envファイルを編集してOPENAI_API_KEYを設定
```

### 2. Docker起動

```bash
docker compose up -d
```

### 3. データ準備

```bash
# 依存関係インストール
uv venv && uv pip install -e .

# データセットダウンロード
python scripts/download_dataset.py

# ドキュメント取り込み
python scripts/ingest_documents.py
```

### 4. アクセス

- **Web UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## アーキテクチャ

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  UI Server  │───▶│  API Server │───▶│  Vector DB  │
│ (Streamlit) │    │  (FastAPI)  │    │  (Qdrant)   │
│   :8501     │    │   :8000     │    │   :6333     │
└─────────────┘    └──────┬──────┘    └─────────────┘
                         │
                         ▼
                  ┌───────────────┐
                  │  OpenAI API   │
                  └───────────────┘
```

## ディレクトリ構成

```
laboro-rag-system/
├── src/                 # ソースコード
│   ├── config/          # 設定管理
│   ├── ingestion/       # データ取り込み
│   ├── retrieval/       # 検索
│   ├── generation/      # 生成
│   ├── rag/             # RAG実装
│   └── evaluation/      # 評価
├── app/
│   ├── api/             # FastAPI
│   └── ui/              # Streamlit
├── scripts/             # 実行スクリプト
├── data/                # データ
├── docs/                # ドキュメント
└── tests/               # テスト
```

## プロジェクトステータス

### 開発終了の経緯

V4実装中、OpenAI APIの使用量が想定を超えたため開発を一時中断しました。V4のコードは実装完了していますが、APIキーが無効化されているため評価は未実施です。

詳細: [docs/PROJECT_WRAP_UP_REPORT.md](docs/PROJECT_WRAP_UP_REPORT.md)

## 将来的な改善構想（V5以降）

### V5の目標: 精度85%超え

以下の高度な技術を導入することで、さらなる精度向上を目指します。

#### 1. Double Prompting（2重プロンプト / Verifier）
- **概要**: AIが回答する前に「確認用プロンプト」を挟み、検証AIが回答の正確性をチェック
- **効果**: ハルシネーションや質問の取り違えを最終出力前に防ぐ
- **実装**: LangChainを活用した検証フロー（LangGraphは現時点で想定外）

#### 2. Multi-Agent Architecture
- **司令塔AI (Router)**: 質問を受け付け、適切な専門エージェントへ振り分け
- **専門エージェント**:
  - **Finance Agent**: 金融データ検索特化、数値計算ツール装備
  - **Food Safety Agent**: 食品ガイドライン検索特化、画像検索優先
  - **Manufacturing Agent**: 製造業ドキュメント検索特化
- **統合**: 各エージェントの回答を司令塔がまとめて最終回答

#### 3. Image to JSON変換 + タグ付け
- 画像内の情報を構造化データ（JSON）として抽出
- カテゴリ・属性・数値を自動タグ付けして検索性を向上

### 重要な考慮事項

#### API呼び出し最適化の必要性
多機能化により**API呼び出しが過多になるリスク**があります。精度が100%に近づいた場合、以下のような**削減フェーズ**が必要になる可能性があります：

- **不要な検証ステップの削除**: 高精度なエージェントでは検証が冗長になる場合がある
- **エージェント統合**: 専門エージェントを汎用エージェントに統合して呼び出し回数を削減
- **キャッシュ活用**: 同一クエリや類似クエリの結果を再利用
- **段階的な検証**: 簡単な質問には軽量な検証、複雑な質問にのみ重厚な検証

**原則**: 精度向上とコスト効率のバランスを取り、過度な複雑化を避ける

### 技術スタックの想定

- **LangChain**: ✅ 想定済み（プロンプト管理、Chain構築）
- **LangGraph**: ❌ 現時点では想定外（必要に応じて将来検討）

詳細なロードマップ: [docs/PROJECT_WRAP_UP_REPORT.md](docs/PROJECT_WRAP_UP_REPORT.md)

## ドキュメント

- [プロジェクト終了報告書](docs/PROJECT_WRAP_UP_REPORT.md) - 開発経緯と将来構想
- [V3改善計画](docs/V3_IMPROVEMENT_PLAN.md) - ハイブリッド検索とリランキング
- [精度改善詳細](docs/ACCURACY_IMPROVEMENTS.md) - V2改善の詳細
- [改善結果レポート](docs/IMPROVEMENT_RESULTS.md) - V2評価結果
- [システム仕様書](docs/system_specification.md)
- [環境構築手順書](docs/setup_guide.md)
- [実行手順書](docs/execution_guide.md)
- [精度評価レポート](docs/evaluation_report.md)

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.11+ |
| パッケージ管理 | uv |
| Webフレームワーク | FastAPI |
| UIフレームワーク | Streamlit |
| ベクトルDB | Qdrant |
| LLM | gpt-5-mini |
| Embedding | text-embedding-3-small |
| コンテナ | Docker + Docker Compose |

## Agentic RAGの定義

本プロジェクトでは、Agentic RAGを以下のように定義しています:

> **LLMエージェントが検索・推論・生成プロセスを自律的に制御し、クエリの複雑さに応じて動的に戦略を変更できるRAGシステム**

### コア機能

1. **Query Analysis**: クエリの複雑さを分析
2. **Adaptive Retrieval**: クエリ書き換え、複数回検索
3. **Multi-step Reasoning**: 複雑な質問を分解
4. **Self-Reflection**: 回答品質を評価・改善

## 評価

```bash
# 評価実行
python scripts/run_evaluation.py

# テスト用（10問のみ）
python scripts/run_evaluation.py --limit 10
```

## ライセンス

本プロジェクトは課題提出用です。

## LLM活用について

本プロジェクトではClaude（Anthropic）を以下の用途で活用しました:
- コード生成・設計支援
- ドキュメント作成
- デバッグ支援
