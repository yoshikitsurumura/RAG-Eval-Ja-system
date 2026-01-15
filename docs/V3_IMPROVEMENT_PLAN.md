# V3 改善計画: ハイブリッド検索 + リランク実装

## 目標
- **現在の精度**: 60% (V2)
- **目標精度**: 70-75%
- **ターゲット**: 画像・表の問題を改善（現在50%）

## 戦略

Gemini 3の提案に基づき、以下の3段階の改善を実装します。

### Phase 1: ハイブリッド検索（Hybrid Search）の実装 ⭐ 最優先

#### 概要
ベクトル検索（意味検索）+ BM25（キーワード検索）を組み合わせ、専門用語・固有名詞の取りこぼしを防ぐ。

#### 問題点
- **現状**: ベクトル検索のみ
  - チャンクが大きい（800文字）ため、ベクトルが「全方位的」になり、検索がぼやける
  - 専門用語（例: ソルベンシー・マージン比率）が正確にヒットしない

#### 解決策
BM25（キーワードベース）を併用し、Reciprocal Rank Fusion (RRF)で統合。

#### 実装内容

**1. 依存ライブラリ追加**
```toml
# pyproject.toml
rank-bm25 = "^0.2.2"  # BM25検索用
```

**2. HybridRetriever実装**
`src/retrieval/retriever.py`の`HybridRetriever`クラスを実装:

```python
class HybridRetriever(RetrieverBase):
    """
    ハイブリッドリトリーバー

    ベクトル検索とBM25（キーワード検索）を組み合わせる。
    Reciprocal Rank Fusion (RRF)で結果を統合。
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedder: EmbedderBase,
        alpha: float = 0.5,  # ベクトル検索の重み（0.0-1.0）
        k: int = 60,  # RRFのパラメータ
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        self.k = k

        # BM25用のコーパス構築（初回のみ）
        self._build_bm25_index()

    def _build_bm25_index(self):
        """全チャンクをBM25インデックスに登録"""
        # Qdrantから全チャンク取得
        # トークナイズ（日本語対応）
        # BM25インデックス構築
        pass

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        ハイブリッド検索

        1. ベクトル検索でtop_k*2件取得
        2. BM25検索でtop_k*2件取得
        3. RRFで統合してtop_k件に絞る
        """
        # 1. ベクトル検索
        vector_results = self._vector_search(query, top_k * 2)

        # 2. BM25検索
        bm25_results = self._bm25_search(query, top_k * 2)

        # 3. RRFで統合
        merged_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k
        )

        return RetrievalResult(results=merged_results)
```

**3. 日本語トークナイザー**
BM25は単語単位で動作するため、日本語の形態素解析が必要:

```python
# 軽量: MeCabなしで動作（Sudachi）
from sudachipy import Dictionary

tokenizer = Dictionary().create()
tokens = [m.surface() for m in tokenizer.tokenize(text)]
```

**4. RRF（Reciprocal Rank Fusion）**
```python
def _reciprocal_rank_fusion(
    self,
    list1: List[SearchResult],
    list2: List[SearchResult],
    top_k: int
) -> List[SearchResult]:
    """
    2つのランキング結果を統合

    RRFスコア = Σ 1 / (k + rank)
    """
    scores = {}

    # ベクトル検索のスコア
    for rank, result in enumerate(list1):
        scores[result.chunk_id] = 1 / (self.k + rank + 1)

    # BM25検索のスコアを加算
    for rank, result in enumerate(list2):
        if result.chunk_id in scores:
            scores[result.chunk_id] += 1 / (self.k + rank + 1)
        else:
            scores[result.chunk_id] = 1 / (self.k + rank + 1)

    # スコア順にソート
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [self._get_chunk(cid) for cid in sorted_ids[:top_k]]
```

#### 期待効果
- **キーワードマッチ率向上**: 専門用語が正確にヒット
- **精度向上**: +3-5%ポイント（推定）
- **特に効果がある問題**: Q2, Q8（画像・表の問題）

---

### Phase 2: リランク（Reranking）の実装 ⭐ 高優先

#### 概要
検索で拾った候補（top_k=20程度）をCross-Encoderで再スコアリングし、上位k件に絞る。

#### 問題点
- **現状**: コサイン類似度のみで順位付け
  - 「関係ありそうだけど違う」ドキュメントが上位に来る
  - LLMに渡すコンテキストの純度が低い

#### 解決策
Cross-Encoderで「質問とチャンクのペア」を直接評価し、関連性スコアを算出。

#### 実装内容

**1. 依存ライブラリ追加**
```toml
# pyproject.toml
sentence-transformers = "^2.2.2"  # リランク用
```

**2. Rerankerクラス実装**
`src/retrieval/reranker.py`（新規作成）:

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """
    検索結果のリランク

    Cross-Encoderで質問とチャンクのペアを直接評価。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        候補をリランク

        Args:
            query: 質問文
            candidates: 検索結果（多め、例: 20件）
            top_k: 最終的に返す件数
        """
        # ペアを作成
        pairs = [(query, c.content) for c in candidates]

        # Cross-Encoderでスコア算出
        scores = self.model.predict(pairs)

        # スコア順にソート
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [c for c, _ in ranked[:top_k]]
```

**3. 統合**
`HybridRetriever`に組み込み:

```python
def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
    # 1. ハイブリッド検索で多めに取得（top_k * 4）
    candidates = self._hybrid_search(query, top_k * 4)

    # 2. リランク
    if self.use_rerank:
        reranked = self.reranker.rerank(query, candidates, top_k)
        return RetrievalResult(results=reranked)

    return RetrievalResult(results=candidates[:top_k])
```

#### 期待効果
- **コンテキスト純度向上**: 上位3件の品質が劇的に改善
- **精度向上**: +3-5%ポイント（推定）
- **Phase 1と合わせて**: +5-10%ポイント（累積）

---

### Phase 3: PDF解析の高度化（表のMarkdown化） 🔺 長期課題

#### 概要
表を構造化されたMarkdown形式に変換してインデックス化。

#### 問題点
- **現状**: pdfplumberで単純抽出
  - 表の構造が崩れる（ヘッダーと値が分離）
  - LLMが表を理解できない

#### 解決策
表をMarkdownの表形式（`| 列1 | 列2 |`）に変換。

#### 実装内容

**1. 表検出・抽出の改善**
`src/ingestion/pdf_parser.py`:

```python
def _extract_table_as_markdown(self, table: List[List[str]]) -> str:
    """
    表をMarkdown形式に変換

    Input:  [["Name", "Value"], ["A", "100"], ["B", "200"]]
    Output: | Name | Value |
            |------|-------|
            | A    | 100   |
            | B    | 200   |
    """
    if not table or len(table) < 2:
        return ""

    # ヘッダー
    header = "| " + " | ".join(table[0]) + " |"
    separator = "|" + "|".join(["---" for _ in table[0]]) + "|"

    # データ行
    rows = [f"| {' | '.join(row)} |" for row in table[1:]]

    return "\n".join([header, separator] + rows)
```

**2. メタデータ追加**
チャンクに「表である」ことを記録:

```python
TextChunk(
    content=markdown_table,
    metadata={
        "type": "table",
        "has_structured_data": True,
    }
)
```

#### 期待効果
- **表問題の改善**: Q8などの正解率向上
- **精度向上**: +5-10%ポイント（推定、表問題のみ）
- **注意**: データ再作成が必要（コスト高）

---

## 実装の優先順位

### 即時実装（今すぐ）
1. ✅ **Phase 1: ハイブリッド検索**
   - コスト: 中（1-2時間）
   - 効果: 高（+3-5%）

2. ✅ **Phase 2: リランク**
   - コスト: 低（30分-1時間）
   - 効果: 高（+3-5%）

### 中期実装（次のイテレーション）
3. **Phase 3: 表のMarkdown化**
   - コスト: 高（データ再作成）
   - 効果: 中（表問題のみ）

---

## 実装手順

### Step 1: 依存関係の追加
```bash
# pyproject.toml に追加
poetry add rank-bm25 sentence-transformers sudachipy sudachidict_core
```

### Step 2: HybridRetriever実装
1. `src/retrieval/retriever.py`の`HybridRetriever`クラスを実装
2. BM25インデックス構築ロジックを追加
3. RRF統合ロジックを実装

### Step 3: Reranker実装
1. `src/retrieval/reranker.py`を新規作成
2. `HybridRetriever`に統合

### Step 4: 評価実行
```bash
# データ再インジェスト（不要、既存データでOK）
# ハイブリッド検索は実行時に動的に動作

# 評価実行
docker compose exec api python scripts/run_evaluation.py --rag-types naive --limit 10
```

### Step 5: 結果分析
- V2: 60% → V3: 70-75%（目標）
- 画像・表問題の改善を確認

---

## パラメータチューニング

### ハイブリッド検索
- `alpha`: ベクトル検索の重み（0.5がデフォルト）
  - 0.7: ベクトル検索重視
  - 0.3: BM25重視

### リランク
- `top_k_candidates`: リランク前の候補数（20-40を推奨）
- `model`: Cross-Encoderモデル
  - 軽量: `ms-marco-MiniLM-L-6-v2`
  - 高精度: `ms-marco-electra-base`

---

## コスト・リスク分析

### 計算コスト
- **BM25インデックス**: 初回構築に数秒（メモリ使用量増加）
- **リランク**: 1質問あたり+0.5-1秒（Cross-Encoder推論）
- **全体**: 許容範囲内

### 実装リスク
- **低**: 既存コードへの影響は最小限
- **HybridRetriever**: 既存のRetrieverインターフェースを実装
- **設定で切り替え可**: `retriever_type="hybrid"`で有効化

---

## 期待される最終結果

| 指標 | V1 | V2 | V3（目標） |
|------|----|----|-----------|
| Overall | 50% | 60% | **70-75%** |
| Paragraph | 50% | 75% | 75-80% |
| Image | 50% | 50% | **60-70%** |
| Table | 50% | 50% | **60-70%** |

---

## 次のステップ

1. 依存ライブラリ追加
2. HybridRetriever実装
3. Reranker実装
4. 評価実行
5. V3結果分析

この計画でv3を実装します。
