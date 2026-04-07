# Compliance Copilot: AI Assistant for NIS2 & GDPR

[cite_start]This project is a prototype of an AI-driven expert system developed as part of my **MBA thesis in Cybersecurity**[cite: 936, 941]. [cite_start]It utilizes **Large Language Models (LLM)** and **Retrieval-Augmented Generation (RAG)** to support software development teams in navigating and interpreting complex regulatory requirements[cite: 937, 966, 1104].



## 🧠 Key Technologies
* **Language:** Python
* [cite_start]**Architecture:** Retrieval-Augmented Generation (RAG) [cite: 961, 1120]
* [cite_start]**AI Integration:** OpenAI/Ollama API for generative responses [cite: 1004, 1282]
* [cite_start]**Vector Database:** Local semantic index for legal document retrieval [cite: 1130, 1148, 1261]
* [cite_start]**Domain Focus:** NIS2 Directive, GDPR compliance, and Risk Management [cite: 936, 958, 959]

## How it Works
1. [cite_start]**Data Engineering:** Ingests regulatory PDFs (NIS2, GDPR) and splits them into semantic chunks using logical structures (Articles/Paragraphs)[cite: 1140, 1248, 1251].
2. [cite_start]**Indexing:** Generates high-dimensional embeddings and stores them in a vector index[cite: 1146, 1155, 1256].
3. [cite_start]**Retrieval:** Performs semantic search to find the most relevant legal articles for a user's technical query[cite: 1131, 1149, 1271].
4. [cite_start]**Generation:** Augments the LLM prompt with retrieved context to provide accurate, citation-backed answers, minimizing hallucinations[cite: 1123, 1125, 1271].



## Project Structure
* `app.py`: Main application orchestrator and backend API.
* `build_index.py`: Data pipeline for document processing and vectorization.
* `data/`: Source regulatory documents and methodologies.
* `index/`: Local storage for the generated vector embeddings.

## Academic & Business Context
[cite_start]Designed to bridge the gap between legal requirements and technical implementation in modern **DevSecOps workflows**[cite: 991, 1106, 1107]. [cite_start]This tool supports the "Shift-Left" approach by providing developers with immediate, evidence-based compliance guidance during the design phase[cite: 1107, 1109].

---
*Developed by Jakub Glončák, MBA | VUT FIT Brno*
