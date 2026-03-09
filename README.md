# The Future of Intelligent Data Interaction
Built a next-generation AI-powered intelligent document and reasoning system — not just a chatbot, but a fully orchestrated AI engine designed for real-time, enterprise-grade decision support.

At its core is a FASTAPI-powered high-performance backend integrated with locally hosted Large Language Models (LLMs) for secure, low-latency reasoning. This ensures complete data control, faster inference, and production-level scalability. But what truly differentiates this system is its Agent-driven architecture — autonomous AI agents capable of multi-step reasoning, retrieval, and action-based decision flows rather than simple prompt-response interactions.

➤ What this Platform Leverages?
The platform leverages advanced Retrieval-Augmented Generation (RAG) pipelines to ground every response in verified data. Retrieved information is intelligently refined using a cross-encoder reranking model (cross-encoder/ms-marco-TinyBERT-L-2-v2), dramatically improving contextual precision before final response generation. The result: high-accuracy, low-hallucination outputs even for complex, deep document queries.

## Modular Agent Orchestration with LangChain and LangGraph
Our orchestration layer is built with LangChain and LangGraph, enabling modular, state-aware conversational flows and scalable agent pipelines. This architecture supports dynamic reasoning paths, multi-document synthesis, and structured workflow execution — making it adaptable across industries from legal and finance to research and enterprise analytics.

## High-Performance Data Infrastructure and Intelligent Queuing
On the infrastructure side, we integrate deeply structured PostgreSQL databases, supporting complex relational modeling (One-to-One, Many-to-Many) and direct database API interactions. To maximize speed and cost-efficiency, RabbitMQ-powered intelligent queuing and caching mechanisms reduce redundant computation and optimize frequently accessed queries.

## Advanced OCR for Searchable Document Text Extraction
The system also incorporates advanced OCR capabilities using Docling Core, transforming scanned PDFs, images, and complex file formats into fully searchable, query-ready intelligence streams. This expands the usable data surface dramatically — turning static documents into interactive knowledge assets.

Built using a powerful Python AI stack — including PyTorch, Transformers, NumPy, and robust logging frameworks — this platform delivers:

✅ Enterprise-grade performance

✅ High recall and contextual accuracy

✅ Multi-format document intelligence

✅ Secure, locally controlled AI reasoning

✅ Scalable, modular architecture

This isn’t just a chatbot.
It’s a scalable AI knowledge engine designed to transform how organizations retrieve, reason over, and interact with their data — in real time.

## Getting Started
### Prerequisites

- Python 3.10+
- Environment variable file

### Installation : 

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --port any_xyz_port --reload
   ```
2. **Run Services**
   ## For Running chat
    ```bash
    python rmq-client\rmq_prompt_client.py
    python rmq-client\publish_prompt.py
    ```
   ## For Embeddings generation
    ```bash
    python rmq-client\rmq_embedding_client.py
    python rmq-sync-embeddings.py
    ```

    



