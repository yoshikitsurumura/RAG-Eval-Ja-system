# 実行手順書

## 1. 事前準備

本手順書を実行する前に、[環境構築手順書](./setup_guide.md)の手順が完了していることを確認してください。

### 1.1 確認事項

- [ ] Docker Composeで全コンテナが起動している
- [ ] `.env`ファイルにOpenAI APIキーが設定されている
- [ ] データセットがダウンロード済み
- [ ] ドキュメントがベクトルDBに取り込み済み

```bash
# コンテナ状態確認
docker compose ps

# 期待される出力
# NAME             STATUS
# laboro-api      Up (healthy)
# laboro-ui       Up
# laboro-qdrant   Up (healthy)
```

---

## 2. Web UIでの対話

### 2.1 UIにアクセス

1. ブラウザで http://localhost:8501 を開く
2. サイドバーでRAGタイプを選択:
   - **Naive RAG（ベースライン）**: シンプルなベクトル検索
   - **Agentic RAG（発展版）**: 自律的な検索戦略

### 2.2 質問の実行

1. テキストエリアに質問を入力
2. 「検索」ボタンをクリック
3. 回答と参照ソースを確認

### 2.3 サンプル質問

| 質問 | 期待される回答のキーワード |
|------|------------------------|
| 生命保険の加入率はどのくらいですか？ | 世帯加入率、89.8% |
| AIのセキュリティリスクについて教えてください | 敵対的攻撃、データ汚染 |
| ものづくり白書の主な内容は何ですか？ | DX、人材確保、サプライチェーン |
| 食品トレーサビリティとは何ですか？ | 生産、加工、流通、追跡 |

---

## 3. APIでの対話

### 3.1 APIドキュメント

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3.2 curlでのクエリ実行

**Naive RAG**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "生命保険の加入率はどのくらいですか？",
    "rag_type": "naive",
    "top_k": 5
  }'
```

**Agentic RAG**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "製造業のDX推進における課題と対策を教えてください",
    "rag_type": "agentic",
    "top_k": 5
  }'
```

### 3.3 Pythonでのクエリ実行

```python
import httpx

# Naive RAG
response = httpx.post(
    "http://localhost:8000/query",
    json={
        "question": "生命保険の加入率はどのくらいですか？",
        "rag_type": "naive",
        "top_k": 5
    }
)
result = response.json()
print(f"回答: {result['answer']}")
print(f"ソース数: {len(result['sources'])}")
```

---

## 4. 評価の実行

### 4.1 評価スクリプトの実行

```bash
# 仮想環境をアクティベート
source .venv/bin/activate  # または .venv\Scripts\activate

# 全データセット（300問）で評価 ※時間がかかります
python scripts/run_evaluation.py

# テスト用に10問だけ評価
python scripts/run_evaluation.py --limit 10

# Naive RAGのみ評価
python scripts/run_evaluation.py --rag-types naive --limit 10

# Agentic RAGのみ評価
python scripts/run_evaluation.py --rag-types agentic --limit 10
```

### 4.2 評価結果の確認

評価結果は `data/evaluation/results/` に保存されます:

- `evaluation_summary_NaiveRAG.json`: Naive RAGのサマリー
- `evaluation_detail_NaiveRAG.csv`: Naive RAGの詳細結果
- `evaluation_summary_AgenticRAG.json`: Agentic RAGのサマリー
- `evaluation_detail_AgenticRAG.csv`: Agentic RAGの詳細結果

**サマリーの内容**:
```json
{
  "rag_name": "NaiveRAG",
  "total_questions": 300,
  "metrics_summary": {
    "accuracy": 0.72,
    "answer_correctness": {"mean": 0.72},
    "llm_judge": {"mean": 0.68}
  },
  "domain_breakdown": {
    "finance": {"total": 60, "correct": 45, "accuracy": 0.75},
    "it": {"total": 60, "correct": 42, "accuracy": 0.70},
    ...
  },
  "type_breakdown": {
    "paragraph": {"total": 142, "correct": 110, "accuracy": 0.77},
    "table": {"total": 83, "correct": 55, "accuracy": 0.66},
    "image": {"total": 75, "correct": 50, "accuracy": 0.67}
  }
}
```

---

## 5. コマンドリファレンス

### 5.1 Docker Compose

```bash
# 起動
docker compose up -d

# 停止
docker compose down

# ログ確認
docker compose logs -f
docker compose logs api -f
docker compose logs ui -f

# 再起動
docker compose restart

# 再ビルド
docker compose build --no-cache
docker compose up -d
```

### 5.2 データ操作

```bash
# データセットダウンロード
python scripts/download_dataset.py

# ドキュメント取り込み
python scripts/ingest_documents.py

# モック埋め込みで取り込み（API呼び出しなし）
python scripts/ingest_documents.py --mock

# チャンクサイズ変更
python scripts/ingest_documents.py --chunk-size 300 --chunk-overlap 50
```

### 5.3 評価

```bash
# 基本的な評価
python scripts/run_evaluation.py

# 件数制限
python scripts/run_evaluation.py --limit 50

# RAGタイプ指定
python scripts/run_evaluation.py --rag-types naive agentic

# メトリクス指定
python scripts/run_evaluation.py --metrics answer_correctness llm_judge
```

---

## 6. トラブルシューティング

### 6.1 「ドキュメントが見つかりません」エラー

```bash
# ベクトルDBにデータがあるか確認
curl http://localhost:6333/collections/laboro_rag

# データがない場合は再取り込み
python scripts/ingest_documents.py
```

### 6.2 「API接続エラー」

```bash
# APIサーバーの状態確認
docker compose logs api

# 再起動
docker compose restart api
```

### 6.3 「タイムアウトエラー」

Agentic RAGは複数のLLM呼び出しを行うため、時間がかかることがあります。

```bash
# タイムアウトを延長（Streamlit）
streamlit run app/ui/streamlit_app.py --server.timeout 120
```

---

## 7. 開発者向け情報

### 7.1 ホットリロード開発

```bash
# APIサーバー（変更を自動検出）
uvicorn app.api.main:app --reload

# Streamlit（変更を自動検出）
streamlit run app/ui/streamlit_app.py
```

### 7.2 テスト実行

```bash
# 全テスト
pytest tests/

# カバレッジ付き
pytest tests/ --cov=src
```

### 7.3 コード品質チェック

```bash
# Ruff（リント・フォーマット）
ruff check src/
ruff format src/

# MyPy（型チェック）
mypy src/
```

---

## 8. 次のステップ

評価が完了したら、[精度評価レポート](./evaluation_report.md)で結果を分析してください。
