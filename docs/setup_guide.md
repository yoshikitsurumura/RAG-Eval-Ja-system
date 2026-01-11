# 環境構築手順書

## 1. 前提条件

### 1.1 必要なソフトウェア

| ソフトウェア | バージョン | 確認コマンド |
|-------------|-----------|-------------|
| Docker | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| Python | 3.11+ | `python --version` |
| uv | 0.4+ | `uv --version` |
| Git | 2.40+ | `git --version` |

### 1.2 システム要件

- **OS**: Windows 10/11, macOS 12+, Ubuntu 20.04+
- **RAM**: 8GB以上（推奨16GB）
- **ストレージ**: 10GB以上の空き容量
- **ネットワーク**: インターネット接続必須（OpenAI API、PDFダウンロード）

---

## 2. インストール手順

### 2.1 リポジトリのクローン

```bash
git clone <repository-url>
cd laboro-rag-system
```

### 2.2 uvのインストール（未インストールの場合）

**Windows (PowerShell)**:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS / Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.3 Python依存関係のインストール

```bash
# 仮想環境作成と依存関係インストール
uv venv
uv pip install -e .

# 開発用依存関係（オプション）
uv pip install -e ".[dev]"
```

### 2.4 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# エディタで編集
# Windows
notepad .env
# macOS / Linux
nano .env
```

`.env`ファイルの内容:
```env
# 必須: 提供されたOpenAI APIキーを設定
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# モデル設定（変更不要）
LLM_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant設定（Docker使用時はデフォルトのまま）
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=laboro_rag

# ログレベル
LOG_LEVEL=INFO
```

---

## 3. Docker環境の構築

### 3.1 Dockerイメージのビルド

```bash
# 全コンテナをビルド
docker compose build

# または個別にビルド
docker compose build api
docker compose build ui
```

### 3.2 コンテナの起動

```bash
# 全コンテナを起動
docker compose up -d

# ログを確認
docker compose logs -f
```

### 3.3 起動確認

| サービス | URL | 確認方法 |
|---------|-----|---------|
| Qdrant | http://localhost:6333/dashboard | ブラウザでアクセス |
| API | http://localhost:8000/health | `{"status": "healthy"}` |
| UI | http://localhost:8501 | ブラウザでアクセス |

```bash
# APIヘルスチェック
curl http://localhost:8000/health

# 期待される応答
# {"status":"healthy","version":"0.1.0"}
```

---

## 4. データセットの準備

### 4.1 データセットのダウンロード

```bash
# 仮想環境をアクティベート
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# ダウンロードスクリプトを実行
python scripts/download_dataset.py
```

これにより以下がダウンロードされます:
- `data/raw/documents.csv`: PDFメタデータ
- `data/raw/pdfs/`: 68個のPDFファイル
- `data/evaluation/qa_dataset.parquet`: 300問のQAデータセット

### 4.2 ドキュメントの取り込み

```bash
# PDFをパースしてベクトルDBに格納
python scripts/ingest_documents.py

# オプション: モック埋め込みを使用（API呼び出しなし、開発用）
python scripts/ingest_documents.py --mock
```

### 4.3 取り込み確認

```bash
# Qdrantダッシュボードで確認
# http://localhost:6333/dashboard

# またはAPIで確認
curl http://localhost:8000/info
```

---

## 5. ローカル開発環境（Docker不使用）

### 5.1 Qdrantのローカル起動

```bash
# Dockerで Qdrantのみ起動
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 5.2 APIサーバーの起動

```bash
# 仮想環境をアクティベート
source .venv/bin/activate  # または .venv\Scripts\activate

# APIサーバー起動
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5.3 UIサーバーの起動

```bash
# 別ターミナルで
source .venv/bin/activate

# Streamlit起動
streamlit run app/ui/streamlit_app.py --server.port 8501
```

---

## 6. トラブルシューティング

### 6.1 Docker関連

**問題**: `docker compose up` でエラー
```bash
# Dockerデーモンが起動しているか確認
docker info

# キャッシュをクリアして再ビルド
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

**問題**: ポートが使用中
```bash
# 使用中のポートを確認
# Windows
netstat -ano | findstr :8000
# Linux/macOS
lsof -i :8000

# プロセスを終了するか、docker-compose.ymlでポートを変更
```

### 6.2 Python/uv関連

**問題**: `uv`コマンドが見つからない
```bash
# パスを確認
which uv  # または where uv (Windows)

# 再インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
# シェルを再起動
```

**問題**: 依存関係のインストールエラー
```bash
# キャッシュをクリア
uv cache clean

# 仮想環境を再作成
rm -rf .venv
uv venv
uv pip install -e .
```

### 6.3 OpenAI API関連

**問題**: API認証エラー
```bash
# 環境変数を確認
echo $OPENAI_API_KEY  # または echo %OPENAI_API_KEY% (Windows)

# .envファイルを確認
cat .env | grep OPENAI_API_KEY
```

**問題**: レート制限エラー
- 少し待ってから再試行
- `--mock`オプションで開発を継続

### 6.4 PDFダウンロード関連

**問題**: 一部PDFのダウンロード失敗
- 政府サイトが一時的にダウンしている可能性
- 後で再実行: `python scripts/download_dataset.py`

---

## 7. 環境のクリーンアップ

```bash
# コンテナ停止・削除
docker compose down

# ボリュームも削除
docker compose down -v

# 仮想環境削除
rm -rf .venv

# ダウンロードしたデータ削除
rm -rf data/raw/pdfs
rm -rf data/processed
```

---

## 8. 次のステップ

環境構築が完了したら、[実行手順書](./execution_guide.md)に進んでください。
