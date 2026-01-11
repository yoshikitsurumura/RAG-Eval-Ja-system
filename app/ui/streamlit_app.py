"""
Streamlit UI

RAGシステムのWebインターフェース
"""

import os

import httpx
import streamlit as st

# API設定
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ページ設定
st.set_page_config(
    page_title="Laboro RAG System",
    page_icon="🔍",
    layout="wide",
)


def query_rag(question: str, rag_type: str, top_k: int) -> dict:
    """RAG APIにクエリを送信"""
    try:
        response = httpx.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "rag_type": rag_type,
                "top_k": top_k,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"APIエラー: {e}")
        return None


def check_api_health() -> bool:
    """APIヘルスチェック"""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def main():
    """メイン処理"""
    # ヘッダー
    st.title("🔍 Laboro RAG System")
    st.markdown(
        """
        日本語RAG評価データセットを使用したRAGシステムです。
        **Naive RAG**（ベースライン）と**Agentic RAG**（発展版）を切り替えて使用できます。
        """
    )

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        # RAGタイプ選択
        rag_type = st.radio(
            "RAGタイプ",
            options=["naive", "agentic"],
            format_func=lambda x: "Naive RAG（ベースライン）"
            if x == "naive"
            else "Agentic RAG（発展版）",
            help="Naive RAG: シンプルなベクトル検索\nAgentic RAG: 自律的な検索戦略",
        )

        # 検索結果数
        top_k = st.slider(
            "検索結果数 (top_k)",
            min_value=1,
            max_value=10,
            value=5,
            help="検索で取得するドキュメント数",
        )

        st.divider()

        # API状態
        st.subheader("📡 API Status")
        if check_api_health():
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
            st.info(f"API URL: {API_URL}")

        st.divider()

        # 情報
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **データソース**: 日本の官公庁・公的機関文書
            - 金融
            - IT
            - 製造業
            - 公共
            - 小売
            """
        )

    # メインコンテンツ
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 質問を入力")

        # 質問入力
        question = st.text_area(
            "質問",
            placeholder="例: 生命保険の加入率はどのくらいですか？",
            height=100,
            label_visibility="collapsed",
        )

        # サンプル質問
        st.caption("サンプル質問:")
        sample_questions = [
            "生命保険の加入率はどのくらいですか？",
            "AIのセキュリティリスクについて教えてください",
            "ものづくり白書の主な内容は何ですか？",
            "食品トレーサビリティとは何ですか？",
        ]
        cols = st.columns(2)
        for i, sq in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(sq, key=f"sample_{i}", use_container_width=True):
                    question = sq
                    st.session_state["question"] = sq

        # 検索ボタン
        if st.button("🔍 検索", type="primary", use_container_width=True):
            if question:
                with st.spinner("検索中..."):
                    result = query_rag(question, rag_type, top_k)

                if result:
                    st.session_state["result"] = result
                    st.session_state["rag_type_used"] = rag_type
            else:
                st.warning("質問を入力してください")

    with col2:
        st.header("📊 RAG情報")
        if rag_type == "naive":
            st.info(
                """
                **Naive RAG**

                シンプルなベクトル検索と回答生成を行うベースラインRAG。

                1. クエリを埋め込み
                2. ベクトル検索
                3. LLMで回答生成
                """
            )
        else:
            st.info(
                """
                **Agentic RAG**

                LLMエージェントが自律的に検索戦略を制御。

                1. クエリ分析
                2. 適応的検索
                3. 回答生成
                4. 自己評価・改善
                """
            )

    # 結果表示
    if "result" in st.session_state and st.session_state["result"]:
        st.divider()
        result = st.session_state["result"]

        # 回答
        st.header("📝 回答")
        st.markdown(f"**使用RAG**: {st.session_state.get('rag_type_used', 'unknown')}")
        st.markdown(result["answer"])

        # ソース
        st.header("📚 参照ソース")
        for i, source in enumerate(result.get("sources", []), start=1):
            with st.expander(
                f"[{i}] {source['source_file']} (p.{source['page_number']}) - スコア: {source['score']:.3f}"
            ):
                st.markdown(source["content"])

        # メタデータ
        with st.expander("🔧 メタデータ"):
            st.json(result.get("metadata", {}))


if __name__ == "__main__":
    main()
