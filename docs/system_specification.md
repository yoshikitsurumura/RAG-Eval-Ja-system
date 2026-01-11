# システム仕様書

## 1. システム概要

### 1.1 目的
本システムは、日本語RAG評価データセット（RAG-Evaluation-Dataset-JA）を知識源とした実践的なRAG（Retrieval-Augmented Generation）システムです。

### 1.2 主要機能
- **Naive RAG**: 単純なベクトル検索と回答生成を行うベースライン実装
- **Agentic RAG**: LLMエージェントが自律的に検索戦略を制御する発展版実装
- **Web UI**: Streamlitベースの対話インターフェース
- **評価パイプライン**: RAGシステムの精度評価機能

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  UI Server  │───▶│  API Server │───▶│  Vector DB  │     │
│  │ (Streamlit) │    │  (FastAPI)  │    │  (Qdrant)   │     │
│  │   :8501     │    │   :8000     │    │   :6333     │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                                │
│                            ▼                                │
│                    ┌───────────────┐                        │
│                    │  OpenAI API   │                        │
│                    │  (External)   │                        │
│                    └───────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 マルチコンテナ構成

| コンテナ | 役割 | ポート |
|---------|------|--------|
| `laboro-ui` | UIサーバー（Streamlit） | 8501 |
| `laboro-api` | アプリケーションサーバー（FastAPI） | 8000 |
| `laboro-qdrant` | データベースサーバー（Qdrant） | 6333, 6334 |

---

## 3. モジュール設計

### 3.1 ディレクトリ構成

```
laboro-rag-system/
├── src/
│   ├── config/          # 設定管理
│   │   └── settings.py  # Pydantic Settings
│   ├── ingestion/       # データ取り込み
│   │   ├── pdf_parser.py    # PDFパーサー
│   │   ├── text_splitter.py # テキスト分割
│   │   └── embedder.py      # 埋め込み生成
│   ├── retrieval/       # 検索
│   │   ├── vector_store.py  # ベクトルストア
│   │   └── retriever.py     # リトリーバー
│   ├── generation/      # 生成
│   │   ├── llm_client.py    # LLMクライアント
│   │   └── prompt_templates.py # プロンプト
│   ├── rag/             # RAG実装
│   │   ├── base.py          # 基底クラス
│   │   ├── naive_rag.py     # Naive RAG
│   │   └── agentic_rag.py   # Agentic RAG
│   └── evaluation/      # 評価
│       ├── metrics.py       # 評価メトリクス
│       └── evaluator.py     # 評価実行
├── app/
│   ├── api/main.py      # FastAPI
│   └── ui/streamlit_app.py  # Streamlit UI
├── scripts/             # 実行スクリプト
├── data/                # データ
└── docs/                # ドキュメント
```

### 3.2 クラス設計（UML）

```
┌───────────────────┐
│     RAGBase       │ <<abstract>>
├───────────────────┤
│ + name: str       │
│ + description: str│
├───────────────────┤
│ + query()         │
│ # _format_context()│
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
┌───┴───┐   ┌───┴───────┐
│NaiveRAG│   │AgenticRAG │
├────────┤   ├───────────┤
│retriever│   │query_analyzer│
│llm_client│  │reflection   │
└────────┘   └───────────┘
```

---

## 4. RAG実装仕様

### 4.1 Naive RAG

#### 処理フロー
1. ユーザークエリを受信
2. クエリを埋め込みベクトルに変換
3. Qdrantでコサイン類似度検索（top_k=5）
4. 検索結果をコンテキストとしてフォーマット
5. gpt-5-miniで回答生成

#### コード例
```python
class NaiveRAG(RAGBase):
    def query(self, question: str) -> RAGResponse:
        # 1. 検索
        retrieval_result = self.retriever.retrieve(question, top_k=5)

        # 2. コンテキスト生成
        context = self._format_context(retrieval_result.results)

        # 3. 回答生成
        response = self.llm_client.generate(
            prompt=f"Context: {context}\nQuestion: {question}",
            system_prompt="コンテキストに基づいて回答してください"
        )

        return RAGResponse(answer=response.content, sources=...)
```

### 4.2 Agentic RAG

#### 独自定義
> **Agentic RAG**: LLMエージェントが検索・推論・生成プロセスを自律的に制御し、クエリの複雑さに応じて動的に戦略を変更できるRAGシステム

#### コア機能

| 機能 | 説明 |
|------|------|
| Query Analysis | クエリの複雑さを分析し、simple/complexを判定 |
| Adaptive Retrieval | クエリ書き換え、複数回検索による精度向上 |
| Multi-step Reasoning | 複雑な質問を分解して段階的に回答 |
| Self-Reflection | 回答品質を1-5で評価し、必要に応じて再生成 |

#### ステートマシン

```
[Start] → QueryAnalysis → [Simple?]
                              │
              ┌───────────────┴───────────────┐
              │ Yes                           │ No
              ▼                               ▼
        SimpleRetrieval              DecomposeQuery
              │                               │
              ▼                               ▼
          Generate ◄──────────────── Synthesis
              │
              ▼
         Reflection → [Quality OK?] → [End]
              │              │ No
              └──────────────┘
```

---

## 5. API仕様

### 5.1 エンドポイント一覧

| Method | Path | 説明 |
|--------|------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/info` | システム情報 |
| POST | `/query` | RAGクエリ実行 |
| POST | `/query/naive` | Naive RAGでクエリ |
| POST | `/query/agentic` | Agentic RAGでクエリ |
| GET | `/rag/types` | 利用可能なRAGタイプ |

### 5.2 クエリリクエスト

```json
{
  "question": "生命保険の加入率はどのくらいですか？",
  "rag_type": "naive",
  "top_k": 5
}
```

### 5.3 クエリレスポンス

```json
{
  "question": "生命保険の加入率はどのくらいですか？",
  "answer": "2021年度の調査によると...",
  "sources": [
    {
      "content": "...",
      "source_file": "生命保険実態調査.pdf",
      "page_number": 5,
      "score": 0.89
    }
  ],
  "rag_type": "naive",
  "metadata": {...}
}
```

---

## 6. 技術スタック

| カテゴリ | 技術 | バージョン |
|---------|------|-----------|
| 言語 | Python | 3.11+ |
| パッケージ管理 | uv | latest |
| Webフレームワーク | FastAPI | 0.115+ |
| UIフレームワーク | Streamlit | 1.40+ |
| ベクトルDB | Qdrant | latest |
| LLMフレームワーク | LangChain | 0.3+ |
| LLM | gpt-5-mini | - |
| Embedding | text-embedding-3-small | - |
| コンテナ | Docker + Docker Compose | - |

---

## 7. 設計原則

### 7.1 オブジェクト指向設計
- **抽象基底クラス**: `RAGBase`, `EmbedderBase`, `RetrieverBase`等
- **依存性注入**: コンストラクタでコンポーネントを注入可能
- **ファクトリーパターン**: `get_rag()`, `get_embedder()`等

### 7.2 SOLID原則の適用
- **S**: 各クラスは単一責任（Parser, Splitter, Embedder等）
- **O**: 基底クラスを継承して拡張可能
- **L**: NaiveRAG/AgenticRAGはRAGBaseと置換可能
- **I**: 必要なインターフェースのみ定義
- **D**: 具象クラスではなく抽象に依存

---

## 8. セキュリティ考慮事項

- APIキーは環境変数で管理（`.env`ファイル）
- `.gitignore`で機密ファイルを除外
- CORSミドルウェアの設定
- 入力バリデーション（Pydantic）

---

## 9. 制限事項・既知の問題

1. **API予算制限**: $30のクレジット制限あり
2. **モデル制限**: gpt-5-miniとtext-embedding-3-smallのみ使用可能
3. **PDFパース**: 複雑なレイアウトのPDFは精度が低下する可能性
4. **画像コンテキスト**: 画像内テキストのOCRは未対応

---

## 10. 今後の拡張可能性

- ハイブリッド検索（ベクトル + BM25）
- マルチモーダルRAG（画像対応）
- ストリーミングレスポンス
- キャッシュ機構
- A/Bテスト機能
