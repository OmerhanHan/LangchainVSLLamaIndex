import os
import time
import tempfile
import traceback
import json
import psutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Tiktoken for token counting ───
import tiktoken


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flask App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LlamaIndex Pipeline  (Ollama + HuggingFace)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_llamaindex_pipeline(tmp_dir: str, query: str, model: str, ollama_url: str):
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

    # Retrieve similar documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Query the LLM with context
    messages = [
        SystemMessage(content="Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Eğer bağlamda cevap yoksa, bilmediğini belirt."),
        HumanMessage(content=f"Bağlam:\n{context}\n\nSoru: {query}")
    ]
    llm_response = llm.invoke(messages)
    t_query = time.perf_counter() - t2
    metrics["query_time"] = round(t_query, 4)

    answer = llm_response.content
    metrics["answer_tokens"] = count_tokens(answer)

    mem_after = get_memory_usage_mb()
    metrics["memory_used_mb"] = round(mem_after - mem_before, 2)
    metrics["total_time"] = round(t_load + t_index + t_query, 4)

    source_docs = []
    for doc in retrieved_docs:
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
#  Flask Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    try:
        files = request.files.getlist("files")
        query = request.form.get("query", "").strip()
        model = request.form.get("model", "llama3.2")
        ollama_url = request.form.get("ollama_url", "http://localhost:11434")
        chunk_size = int(request.form.get("chunk_size", 1000))
        chunk_overlap = int(request.form.get("chunk_overlap", 200))

        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "Lütfen en az bir dosya yükleyin!"}), 400
        if not query:
            return jsonify({"error": "Lütfen bir soru yazın!"}), 400

        # Save files to temp dir
        tmp_dir = tempfile.mkdtemp()
        file_paths = []
        file_names = []
        for f in files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(tmp_dir, filename)
                f.save(path)
                file_paths.append(path)
                file_names.append(f.filename)

        if not file_paths:
            return jsonify({"error": "Desteklenen dosya formatı bulunamadı (PDF, TXT, MD)!"}), 400

        results = {"files": file_names, "query": query, "model": model}

        # ── Run LlamaIndex ──
        try:
            results["llamaindex"] = run_llamaindex_pipeline(tmp_dir, query, model, ollama_url)
        except Exception as e:
            results["llamaindex"] = {"error": str(e), "traceback": traceback.format_exc()}

        # ── Run LangChain ──
        try:
            results["langchain"] = run_langchain_pipeline(
                file_paths, query, model, ollama_url, chunk_size, chunk_overlap
            )
        except Exception as e:
            results["langchain"] = {"error": str(e), "traceback": traceback.format_exc()}

        # ── Calculate winner ──
        if "metrics" in results.get("llamaindex", {}) and "metrics" in results.get("langchain", {}):
            lm = results["llamaindex"]["metrics"]
            lcm = results["langchain"]["metrics"]
            comparisons = [
                ("total_time", True), ("load_time", True), ("index_time", True),
                ("query_time", True), ("num_chunks", False), ("memory_used_mb", True),
                ("answer_tokens", False),
            ]
            llama_wins = 0
            lang_wins = 0
            for key, lower_better in comparisons:
                vl, vc = lm.get(key, 0), lcm.get(key, 0)
                if lower_better:
                    if vl < vc: llama_wins += 1
                    elif vc < vl: lang_wins += 1
                else:
                    if vl > vc: llama_wins += 1
                    elif vc > vl: lang_wins += 1
            results["summary"] = {
                "llama_wins": llama_wins,
                "lang_wins": lang_wins,
                "ties": len(comparisons) - llama_wins - lang_wins,
            }

        results["timestamp"] = datetime.now().isoformat()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
