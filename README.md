# 🔬 LlamaIndex vs LangChain — Test Lab

Yerel Ollama modelleri kullanarak **LlamaIndex** ve **LangChain** framework'lerini karşılaştıran bir RAG (Retrieval-Augmented Generation) test uygulaması. Tamamen ücretsiz ve yerel olarak çalışır — API key gerekmez.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Mimari](#-mimari)
- [Gereksinimler](#-gereksinimler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Karşılaştırılan Metrikler](#-karşılaştırılan-metrikler)
- [Proje Yapısı](#-proje-yapısı)
- [Ekran Görüntüleri](#-ekran-görüntüleri)

---

## ✨ Özellikler

- 📄 **Doküman yükleme** — PDF, TXT, MD formatlarını destekler (drag & drop)
- 🦙 **LlamaIndex Pipeline** — HuggingFace embedding + in-memory vektör indeks
- 🦜 **LangChain Pipeline** — Ollama embedding + ChromaDB vektör veritabanı
- 📊 **7 farklı metrikte** yan yana karşılaştırma
- 📈 **Görsel grafikler** — Süre karşılaştırma bar chart'ları
- 💡 **Cevap karşılaştırması** — Her iki framework'ün cevabı yan yana
- 📚 **Kaynak gösterimi** — Hangi doküman parçalarının referans alındığı
- 📥 **JSON export** — Sonuçları dışa aktarma
- 🔒 **Tamamen yerel** — Verileriniz bilgisayarınızdan çıkmaz

---

## 🏗 Mimari

```
                    ┌─────────────┐
                    │  Flask UI   │
                    │  (Browser)  │
                    └──────┬──────┘
                           │ AJAX POST /compare
                    ┌──────▼──────┐
                    │  Flask API  │
                    │   app.py    │
                    └──┬──────┬───┘
           ┌───────────┘      └───────────┐
    ┌──────▼──────┐              ┌────────▼───────┐
    │ LlamaIndex  │              │   LangChain    │
    │  Pipeline   │              │   Pipeline     │
    ├─────────────┤              ├────────────────┤
    │ HuggingFace │              │ Ollama         │
    │ Embedding   │              │ Embedding      │
    │ (MiniLM-L6) │              │                │
    ├─────────────┤              ├────────────────┤
    │ In-Memory   │              │ ChromaDB       │
    │ VectorStore │              │ VectorStore    │
    └──────┬──────┘              └────────┬───────┘
           │                              │
           └──────────┬───────────────────┘
                      │
               ┌──────▼──────┐
               │   Ollama    │
               │  (Local LLM)│
               └─────────────┘
```

---

## 📦 Gereksinimler

- **Python** 3.11+
- **Ollama** (yerel LLM sunucusu)
- ~2GB disk alanı (model + bağımlılıklar)
- 8GB+ RAM önerilir

---

## 🚀 Kurulum

### 1. Ollama Kurulumu

```bash
# macOS
brew install ollama

# veya https://ollama.com adresinden indirin
```

### 2. Model İndirme

```bash
# Varsayılan model
ollama pull llama3.2

# Diğer seçenekler
ollama pull mistral
ollama pull gemma2
ollama pull phi3
```

### 3. Python Bağımlılıkları

```bash
# Sanal ortam oluşturun
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

---

## 🎯 Kullanım

### 1. Ollama'yı Başlatın

```bash
ollama serve
```

### 2. Uygulamayı Çalıştırın

```bash
source venv/bin/activate
python app.py
```

### 3. Tarayıcıda Açın

```
http://localhost:5050
```

### 4. Test Edin

1. 📄 Bir veya birden fazla doküman yükleyin (PDF, TXT, MD)
2. 💬 Doküman hakkında bir soru yazın
3. ⚙️ İsterseniz model ve chunk ayarlarını değiştirin
4. 🚀 **Karşılaştır** butonuna tıklayın
5. 📊 Sonuçları inceleyin ve JSON olarak indirin

---

## 📊 Karşılaştırılan Metrikler

| Metrik | Açıklama | Kazanan Kriteri |
|--------|----------|-----------------|
| ⏱ **Toplam Süre** | Tüm pipeline süresi | Düşük → iyi |
| 📥 **Yükleme Süresi** | Dokümanları okuma süresi | Düşük → iyi |
| 🗂 **İndeksleme Süresi** | Chunk + vektör oluşturma | Düşük → iyi |
| 🔍 **Sorgu Süresi** | Arama + LLM yanıt süresi | Düşük → iyi |
| 📦 **Chunk Sayısı** | Doküman parça sayısı | Yüksek → iyi |
| 💾 **Bellek Kullanımı** | RAM tüketimi (MB) | Düşük → iyi |
| 🔢 **Cevap Token** | Üretilen yanıt uzunluğu | Yüksek → iyi |

### Framework Farkları

| | LlamaIndex | LangChain |
|--|-----------|-----------|
| **Embedding** | HuggingFace (all-MiniLM-L6-v2) | Ollama Embeddings |
| **Vektör DB** | Bellek içi (in-memory) | ChromaDB (disk) |
| **Chunking** | Otomatik | Manuel (ayarlanabilir) |
| **LLM** | Ollama | Ollama |

---

## 📁 Proje Yapısı

```
test/
├── app.py                 # Flask backend + pipeline'lar
├── templates/
│   └── index.html         # Web arayüzü (HTML/CSS/JS)
├── requirements.txt       # Python bağımlılıkları
├── .python-version        # Python sürümü (3.11)
├── .gitignore
└── venv/                  # Sanal ortam (git'e dahil değil)
```

---

## ⚙️ Yapılandırma

Tüm ayarlar web arayüzünden değiştirilebilir:

| Ayar | Varsayılan | Açıklama |
|------|-----------|----------|
| Ollama URL | `http://localhost:11434` | Ollama sunucu adresi |
| Model | `llama3.2` | Kullanılacak LLM modeli |
| Chunk Boyutu | `1000` | LangChain metin parçalama boyutu |
| Chunk Overlap | `200` | Parçalar arası örtüşme |

---

## 🐛 Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `Connection refused` | Ollama çalışıyor mu kontrol edin: `ollama serve` |
| `Model not found` | Modeli indirin: `ollama pull llama3.2` |
| Port 5000 kullanımda | macOS AirPlay sorunu — uygulama port 5050 kullanıyor |
| Yavaş yanıt | Daha küçük model deneyin (ör. `phi3`) veya doküman boyutunu azaltın |

---

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlıdır.

---

<div align="center">
  <strong>🔬 LlamaIndex vs LangChain Test Lab</strong><br>
  Ücretsiz & Yerel — Ollama ile
</div>
