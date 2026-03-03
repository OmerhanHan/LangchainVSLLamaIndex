import streamlit as st
import os
import time
import tempfile
import traceback
import json
import psutil
from datetime import datetime

# ─── LlamaIndex Imports ───
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings as LlamaSettings,
)
from llama_index.llms.ollama import Ollama as LlamaOllama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ─── LangChain Imports ───
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ─── Tiktoken for token counting ───
import tiktoken


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string (approximate)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def save_uploaded_files(uploaded_files, tmp_dir: str) -> list[str]:
    """Save uploaded files to a temporary directory and return paths."""
    paths = []
    for uf in uploaded_files:
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LlamaIndex Pipeline  (Ollama + HuggingFace)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_llamaindex_pipeline(tmp_dir: str, query: str, model: str, ollama_url: str):
    """
    Run the LlamaIndex pipeline: load → index → query.
    Uses Ollama (free, local) for LLM and HuggingFace for embeddings.
    """
    metrics = {}
    mem_before = get_memory_usage_mb()

    # ── 1. Document Loading ──
    t0 = time.perf_counter()
    LlamaSettings.llm = LlamaOllama(
        model=model, base_url=ollama_url, request_timeout=120.0
    )
    LlamaSettings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = SimpleDirectoryReader(tmp_dir).load_data()
    t_load = time.perf_counter() - t0
    metrics["load_time"] = round(t_load, 4)
    metrics["num_documents"] = len(documents)

    total_text = " ".join([d.text for d in documents])
    metrics["total_chars"] = len(total_text)
    metrics["total_tokens_input"] = count_tokens(total_text)

    # ── 2. Indexing ──
    t1 = time.perf_counter()
    index = VectorStoreIndex.from_documents(documents)
    t_index = time.perf_counter() - t1
    metrics["index_time"] = round(t_index, 4)
    metrics["num_chunks"] = len(documents)

    # ── 3. Querying ──
    t2 = time.perf_counter()
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    t_query = time.perf_counter() - t2
    metrics["query_time"] = round(t_query, 4)

    answer = str(response)
    metrics["answer_tokens"] = count_tokens(answer)

    mem_after = get_memory_usage_mb()
    metrics["memory_used_mb"] = round(mem_after - mem_before, 2)
    metrics["total_time"] = round(t_load + t_index + t_query, 4)

    source_nodes = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            source_nodes.append(
                {
                    "score": round(node.score, 4) if node.score else None,
                    "text_preview": node.text[:200] + "..."
                    if len(node.text) > 200
                    else node.text,
                }
            )
    metrics["source_nodes"] = source_nodes

    return {"answer": answer, "metrics": metrics}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LangChain Pipeline  (Ollama + Ollama Embeddings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_langchain_pipeline(
    file_paths: list[str],
    query: str,
    model: str,
    ollama_url: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Run the LangChain pipeline: load → split → vectorize → query.
    Uses Ollama (free, local) for LLM and embeddings.
    """
    metrics = {}
    mem_before = get_memory_usage_mb()

    # ── 1. Document Loading ──
    t0 = time.perf_counter()
    all_docs = []
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(fp)
        else:
            loader = TextLoader(fp, encoding="utf-8")
        all_docs.extend(loader.load())
    t_load = time.perf_counter() - t0
    metrics["load_time"] = round(t_load, 4)
    metrics["num_documents"] = len(all_docs)

    total_text = " ".join([d.page_content for d in all_docs])
    metrics["total_chars"] = len(total_text)
    metrics["total_tokens_input"] = count_tokens(total_text)

    # ── 2. Splitting & Indexing ──
    t1 = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)
    metrics["num_chunks"] = len(chunks)

    embeddings = OllamaEmbeddings(model=model, base_url=ollama_url)
    chroma_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_dir
    )
    t_index = time.perf_counter() - t1
    metrics["index_time"] = round(t_index, 4)

    # ── 3. Querying ──
    t2 = time.perf_counter()
    llm = ChatOllama(model=model, base_url=ollama_url)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    t_query = time.perf_counter() - t2
    metrics["query_time"] = round(t_query, 4)

    answer = result["result"]
    metrics["answer_tokens"] = count_tokens(answer)

    mem_after = get_memory_usage_mb()
    metrics["memory_used_mb"] = round(mem_after - mem_before, 2)
    metrics["total_time"] = round(t_load + t_index + t_query, 4)

    source_docs = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            source_docs.append(
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "page": doc.metadata.get("page", "N/A"),
                    "text_preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
            )
    metrics["source_docs"] = source_docs

    return {"answer": answer, "metrics": metrics}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Streamlit UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    st.set_page_config(
        page_title="🔬 LlamaIndex vs LangChain — Test Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { font-family: 'Inter', sans-serif; }

        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        .main-header h1 { margin: 0; font-weight: 700; font-size: 2rem; }
        .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; }

        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            color: white;
            transition: transform 0.2s;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-card .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }
        .metric-card .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .answer-box {
            background: #0d1117;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            color: #e6edf3;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .winner-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .winner-badge.green {
            background: rgba(0, 212, 170, 0.15);
            color: #00d4aa;
            border: 1px solid rgba(0, 212, 170, 0.3);
        }
        .winner-badge.red {
            background: rgba(255, 107, 107, 0.15);
            color: #ff6b6b;
            border: 1px solid rgba(255, 107, 107, 0.3);
        }

        .source-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            color: #c9d1d9;
        }

        .info-box {
            background: linear-gradient(135deg, #1e3a5f 0%, #1a1a2e 100%);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 1.2rem;
            color: #a8c8ff;
            font-size: 0.9rem;
            margin: 1rem 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Header ──
    st.markdown(
        """
    <div class="main-header">
        <h1>🔬 LlamaIndex vs LangChain — Test Lab</h1>
        <p>Ücretsiz & Yerel — Ollama ile dokümanlarınızı karşılaştırın</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")

        st.markdown(
            """
        <div class="info-box">
            💡 <strong>Tamamen ücretsiz!</strong><br>
            Ollama ile yerel bilgisayarınızda çalışır.<br>
            API key gerekmez.
        </div>
        """,
            unsafe_allow_html=True,
        )

        ollama_url = st.text_input(
            "🌐 Ollama URL",
            value="http://localhost:11434",
            help="Ollama sunucu adresi",
        )

        model = st.selectbox(
            "🤖 Ollama Model",
            ["llama3.2", "llama3.1", "llama3", "mistral", "gemma2", "phi3", "qwen2.5"],
            index=0,
            help="Yerel olarak çalışacak LLM modeli (önce 'ollama pull model_adi' ile indirin)",
        )

        st.markdown("---")
        st.markdown("### 📐 LangChain Ayarları")

        chunk_size = st.slider(
            "Chunk Boyutu",
            min_value=200,
            max_value=4000,
            value=1000,
            step=100,
            help="Doküman parçalama boyutu (karakter)",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Parçalar arası örtüşme miktarı",
        )

        st.markdown("---")
        st.markdown(
            """
        ### 📋 Nasıl Kullanılır
        1. Ollama'yı kurun → `brew install ollama`
        2. Model indirin → `ollama pull llama3.2`
        3. Ollama'yı başlatın → `ollama serve`
        4. Doküman yükleyin
        5. Soru yazın → **Karşılaştır** 🚀
        """
        )

    # ── Main Content ──
    col_upload, col_query = st.columns([1, 1])

    with col_upload:
        st.markdown("### 📂 Doküman Yükleme")
        uploaded_files = st.file_uploader(
            "Dosyalarınızı sürükleyin veya seçin",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="PDF, TXT veya MD formatında dosyalar yükleyin",
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} dosya yüklendi")
            for uf in uploaded_files:
                size_kb = len(uf.getvalue()) / 1024
                st.markdown(f"  📄 **{uf.name}** — `{size_kb:.1f} KB`")

    with col_query:
        st.markdown("### 💬 Soru Sorun")
        query = st.text_area(
            "Dokümanlar hakkında sorunuzu yazın",
            placeholder="Örnek: Bu doküman ne hakkında? Temel konuları özetle.",
            height=120,
        )

    st.markdown("---")

    # ── Run button ──
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_button = st.button(
            "🚀 Karşılaştır — Her İki Framework ile İşle",
            use_container_width=True,
            type="primary",
        )

    if run_button:
        # Validation
        if not uploaded_files:
            st.error("❌ Lütfen en az bir dosya yükleyin!")
            return
        if not query.strip():
            st.error("❌ Lütfen bir soru yazın!")
            return

        # Save files to temp dir
        tmp_dir = tempfile.mkdtemp()
        file_paths = save_uploaded_files(uploaded_files, tmp_dir)

        st.markdown("---")

        # ── Run Pipelines ──
        llama_result = None
        langchain_result = None

        col_l, col_r = st.columns(2)

        # ── LlamaIndex ──
        with col_l:
            st.markdown(
                "### <span style='color:#00d4aa'>🦙 LlamaIndex</span>",
                unsafe_allow_html=True,
            )
            with st.spinner("LlamaIndex işliyor..."):
                try:
                    llama_result = run_llamaindex_pipeline(
                        tmp_dir, query, model, ollama_url
                    )
                    st.success("✅ LlamaIndex tamamlandı!")
                except Exception as e:
                    st.error(f"❌ LlamaIndex hatası: {e}")
                    st.code(traceback.format_exc())

        # ── LangChain ──
        with col_r:
            st.markdown(
                "### <span style='color:#ff6b6b'>🦜 LangChain</span>",
                unsafe_allow_html=True,
            )
            with st.spinner("LangChain işliyor..."):
                try:
                    langchain_result = run_langchain_pipeline(
                        file_paths, query, model, ollama_url,
                        chunk_size, chunk_overlap,
                    )
                    st.success("✅ LangChain tamamlandı!")
                except Exception as e:
                    st.error(f"❌ LangChain hatası: {e}")
                    st.code(traceback.format_exc())

        # ── Results ──
        if llama_result and langchain_result:
            st.markdown("---")
            st.markdown("## 📊 Karşılaştırma Sonuçları")

            lm = llama_result["metrics"]
            lcm = langchain_result["metrics"]

            # ── Metric comparison cards ──
            def winner_label(val_l, val_c, lower_better=True):
                if lower_better:
                    if val_l < val_c:
                        return "🦙", "green"
                    elif val_c < val_l:
                        return "🦜", "red"
                    else:
                        return "🤝", "green"
                else:
                    if val_l > val_c:
                        return "🦙", "green"
                    elif val_c > val_l:
                        return "🦜", "red"
                    else:
                        return "🤝", "green"

            metrics_data = [
                ("⏱ Toplam Süre", lm["total_time"], lcm["total_time"], "s", True),
                ("📥 Yükleme Süresi", lm["load_time"], lcm["load_time"], "s", True),
                ("🗂 İndeksleme Süresi", lm["index_time"], lcm["index_time"], "s", True),
                ("🔍 Sorgu Süresi", lm["query_time"], lcm["query_time"], "s", True),
                ("📦 Chunk Sayısı", lm["num_chunks"], lcm["num_chunks"], "", False),
                ("💾 Bellek (MB)", lm["memory_used_mb"], lcm["memory_used_mb"], "MB", True),
                ("🔢 Cevap Token", lm["answer_tokens"], lcm["answer_tokens"], "", False),
            ]

            for i in range(0, len(metrics_data), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(metrics_data):
                        break
                    label, val_l, val_c, unit, lower_better = metrics_data[idx]
                    icon, color = winner_label(val_l, val_c, lower_better)
                    with col:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div style="display:flex; justify-content:space-around; align-items:center; margin-top:0.5rem;">
                                <div>
                                    <div style="font-size:0.7rem; color:#00d4aa;">🦙 LlamaIndex</div>
                                    <div class="metric-value" style="color:#00d4aa;">{val_l}{unit}</div>
                                </div>
                                <div style="font-size:1.2rem; opacity:0.3;">vs</div>
                                <div>
                                    <div style="font-size:0.7rem; color:#ff6b6b;">🦜 LangChain</div>
                                    <div class="metric-value" style="color:#ff6b6b;">{val_c}{unit}</div>
                                </div>
                            </div>
                            <div style="margin-top:0.5rem;">
                                <span class="winner-badge {color}">Kazanan: {icon}</span>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bar charts ──
            st.markdown("### 📈 Süre Karşılaştırması")

            import pandas as pd

            time_data = pd.DataFrame(
                {
                    "Aşama": ["Yükleme", "İndeksleme", "Sorgu", "Toplam"],
                    "🦙 LlamaIndex (s)": [
                        lm["load_time"], lm["index_time"],
                        lm["query_time"], lm["total_time"],
                    ],
                    "🦜 LangChain (s)": [
                        lcm["load_time"], lcm["index_time"],
                        lcm["query_time"], lcm["total_time"],
                    ],
                }
            )
            st.bar_chart(time_data.set_index("Aşama"), color=["#00d4aa", "#ff6b6b"])

            st.markdown("---")

            # ── Answers side by side ──
            st.markdown("### 💡 Cevapların Karşılaştırması")
            ans_col1, ans_col2 = st.columns(2)

            with ans_col1:
                st.markdown(
                    "#### <span style='color:#00d4aa'>🦙 LlamaIndex Cevabı</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="answer-box">{llama_result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

            with ans_col2:
                st.markdown(
                    "#### <span style='color:#ff6b6b'>🦜 LangChain Cevabı</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="answer-box">{langchain_result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Source Documents ──
            st.markdown("### 📚 Kaynak Dokümanlar")
            src_col1, src_col2 = st.columns(2)

            with src_col1:
                st.markdown("**🦙 LlamaIndex Kaynakları**")
                if lm.get("source_nodes"):
                    for i, node in enumerate(lm["source_nodes"], 1):
                        score_str = f" (skor: {node['score']})" if node["score"] else ""
                        st.markdown(
                            f'<div class="source-card"><strong>Kaynak {i}{score_str}</strong><br>{node["text_preview"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Kaynak bilgisi mevcut değil")

            with src_col2:
                st.markdown("**🦜 LangChain Kaynakları**")
                if lcm.get("source_docs"):
                    for i, doc in enumerate(lcm["source_docs"], 1):
                        source_str = f" ({doc['source']}, sayfa: {doc['page']})"
                        st.markdown(
                            f'<div class="source-card"><strong>Kaynak {i}{source_str}</strong><br>{doc["text_preview"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Kaynak bilgisi mevcut değil")

            st.markdown("---")

            # ── Summary Table ──
            st.markdown("### 📋 Detaylı Özet Tablosu")

            summary_df = pd.DataFrame(
                {
                    "Metrik": [
                        "Yükleme Süresi (s)", "İndeksleme Süresi (s)",
                        "Sorgu Süresi (s)", "Toplam Süre (s)",
                        "Chunk Sayısı", "Toplam Karakter",
                        "Input Token", "Cevap Token",
                        "Bellek Kullanımı (MB)",
                    ],
                    "🦙 LlamaIndex": [
                        lm["load_time"], lm["index_time"],
                        lm["query_time"], lm["total_time"],
                        lm["num_chunks"], lm["total_chars"],
                        lm["total_tokens_input"], lm["answer_tokens"],
                        lm["memory_used_mb"],
                    ],
                    "🦜 LangChain": [
                        lcm["load_time"], lcm["index_time"],
                        lcm["query_time"], lcm["total_time"],
                        lcm["num_chunks"], lcm["total_chars"],
                        lcm["total_tokens_input"], lcm["answer_tokens"],
                        lcm["memory_used_mb"],
                    ],
                }
            )

            lower_better_flags = [True, True, True, True, False, False, False, False, True]
            winners = []
            for idx, row in summary_df.iterrows():
                vl = row["🦙 LlamaIndex"]
                vc = row["🦜 LangChain"]
                lb = lower_better_flags[idx]
                if lb:
                    winners.append("🦙" if vl < vc else ("🦜" if vc < vl else "🤝"))
                else:
                    winners.append("🦙" if vl > vc else ("🦜" if vc > vl else "🤝"))
            summary_df["Kazanan"] = winners

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ── Overall Winner ──
            llama_wins = winners.count("🦙")
            langchain_wins = winners.count("🦜")
            ties = winners.count("🤝")

            st.markdown("---")
            ov1, ov2, ov3 = st.columns(3)
            with ov1:
                st.metric("🦙 LlamaIndex Kazandığı", f"{llama_wins} metrik")
            with ov2:
                st.metric("🦜 LangChain Kazandığı", f"{langchain_wins} metrik")
            with ov3:
                st.metric("🤝 Berabere", f"{ties} metrik")

            if llama_wins > langchain_wins:
                st.success(
                    f"🏆 **Genel Kazanan: 🦙 LlamaIndex** — {llama_wins}/{len(winners)} metrikte daha iyi!"
                )
            elif langchain_wins > llama_wins:
                st.success(
                    f"🏆 **Genel Kazanan: 🦜 LangChain** — {langchain_wins}/{len(winners)} metrikte daha iyi!"
                )
            else:
                st.info("🤝 **Berabere!** Her iki framework da eşit performans gösterdi.")

            # ── Export results ──
            st.markdown("---")
            st.markdown("### 💾 Sonuçları Dışa Aktar")

            export_data = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "query": query,
                "files": [uf.name for uf in uploaded_files],
                "llamaindex": llama_result,
                "langchain": langchain_result,
                "overall_winner": "LlamaIndex"
                if llama_wins > langchain_wins
                else ("LangChain" if langchain_wins > llama_wins else "Tie"),
            }
            st.download_button(
                "📥 JSON olarak indir",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; opacity:0.5; font-size:0.8rem; padding:1rem;">
        🔬 LlamaIndex vs LangChain Test Lab — Ücretsiz & Yerel (Ollama)
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
