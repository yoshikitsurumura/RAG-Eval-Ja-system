# Laboro RAG System

株式会社Laboro.AI LLMエンジニア コーディング課題 - RAGシステムの実装

## 概要

日本語RAG評価データセット（RAG-Evaluation-Dataset-JA）を知識源とした実践的なRAGシステムです。

### 実装内容

- **Naive RAG**: 単純なベクトル検索と回答生成を行うベースライン実装
- **Agentic RAG**: LLMエージェントが自律的に検索戦略を制御する発展版実装

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

## ドキュメント

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
