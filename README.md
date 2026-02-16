# LEO Optima: The Ultimate AI Token Optimization Engine

**LEO Optima** is a high-performance, model-agnostic optimization layer designed to slash LLM API costs by up to 80% without sacrificing output quality. By combining advanced semantic caching, intelligent context management, and a dual-tier deployment model, LEO Optima makes running large-scale AI agents and applications economically sustainable.

---

## 🚀 What's New in v1.1 (TruthOptima Update)

The latest update brings the **TruthOptima** hybrid system into the core of LEO Optima, focusing on **Universal API Compatibility** and **Byzantine Verification**.

- **Universal API Interface**: A new abstraction layer (`api_interfaces.py`) that supports any LLM API in the world.
- **OpenAI Compatibility**: Native support for OpenAI-compatible endpoints, Anthropic, and local LLMs.
- **Byzantine Consensus**: Multi-model verification for high-risk queries to ensure factual accuracy and detect outliers.
- **Enhanced Smart Routing**: Automatically switches between Cache, Fast (single model), and Consensus (multi-model) routes based on query risk, novelty, and coherence.

---

## 🛠 Core Technologies

LEO Optima is built on the mathematical foundations of the **Truth-Optima** hybrid system:

1.  **HNSW Semantic Cache:** A local, high-speed vector database that stores previous successful completions.
2.  **Micro-Memory Influence:** A contextual memory system that enriches embeddings with local history.
3.  **Novelty & Coherence Engines:**
    *   **Novelty:** Measures how "new" or "different" a query is.
    *   **Coherence:** Uses ADMM-based consensus to measure structural complexity.
4.  **Universal API Layer:** Interface-driven design allowing seamless integration of any model provider.

---

## 📦 Repository Structure

- `Truth_Optima.py`: Main system logic and routing engine.
- `api_interfaces.py`: Universal API abstraction layer (v1.1).
- `main.py`: Simple CLI entry point for the system.
- `.env.example`: Template for environment variables.
- `README.md`: Project documentation and roadmap.

---

## 🚀 Getting Started

### Installation
```bash
# Install dependencies
sudo apt-get install -y build-essential python3-dev
sudo pip3 install scikit-learn scipy hnswlib openai
```

### Usage
Run the interactive CLI:
```bash
python3 main.py --interactive --sim
```

Or ask a single question:
```bash
python3 main.py "What is machine learning?" --sim
```

---

## 🗺 Roadmap & Development Plan

### Phase 1: Foundation (COMPLETED)
*   [x] Build a flexible Python Proxy compatible with OpenAI/Anthropic API formats.
*   [x] Integrate the HNSW-based Semantic Cache.
*   [x] Implement Novelty/Coherence scoring for routing logic.
*   [x] **v1.1 Update**: Universal API Interface and Byzantine Consensus.

### Phase 2: Integration & Ecosystem (IN PROGRESS)
*   [ ] Create "One-Click" setup guides for popular platforms like **Open Claw**.
*   [ ] Develop a CLI tool for easy installation and license activation.
*   [ ] Build a local dashboard to track "Dollars Saved" and "Tokens Optimized."

### Phase 3: The Managed Cloud
*   [ ] Deploy the LEO Managed API gateway with multi-tenant support.
*   [ ] Implement a global, shared (but anonymized) semantic cache.

---

## 📄 License & Terms

LEO Optima is a proprietary optimization engine. For licensing inquiries or to join the beta, please visit [leo-optima.com](https://leo-optima.com).

---

*Built with ❤️ for the AI Developer Community.*
