# OmniStudy: Local LLM AI Academic Study Assistant

**Student:** Rhydham Shukla  
**Course:** Machine Learning August 2026 Batch, Unlox Academy  
**Project:** Minor Project 5 (Ollama and Streamlit)  

## Project Overview
OmniStudy is an interactive web based academic study assistant designed to help Machine Learning students review core concepts, verify mathematical derivations, and debug algorithms.

The application runs completely locally on the host machine using Ollama and Streamlit. It connects directly to a quantized DeepSeek R1 reasoning model (deepseek-r1:7b) accelerated by the local NVIDIA GeForce RTX 5050 Laptop GPU, guaranteeing 100% data privacy and zero cloud API charges.

## System Architecture
* **Frontend:** Streamlit 1.66 with custom responsive CSS, dynamic sidebar controls, and chat history management.
* **Backend Inference:** Local Ollama daemon communicating over REST API (http://localhost:11434).
* **Foundation Model:** deepseek-r1:7b (7 billion parameter reasoning model with Chain of Thought reasoning).
* **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM) running CUDA acceleration.

## Key Features
1. **Model Health Detection:** Automatically queries the local Ollama tags endpoint to check daemon status and list available local models.
2. **Chain of Thought Parsing:** Uses regex parsing to separate internal reasoning steps inside think tags, placing them into an interactive UI expander so the main chat stays clean and readable.
3. **Curriculum Quick Prompts:** Preloaded buttons for core Machine Learning concepts (Overfitting, Sigmoid function, Random Forests, Stratified splits, and Outlier capping).
4. **Latency Tracking:** Measures exact end to end inference time and displays a badge below every assistant response.

## Empirical 5 Prompt Benchmark Results (deepseek-r1:7b on RTX 5050)
* Prompt 1 (Overfitting vs Underfitting): 19.73s latency
* Prompt 2 (Sigmoid in Logistic Regression): 25.14s latency
* Prompt 3 (Random Forest vs Decision Trees): 19.11s latency
* Prompt 4 (Stratified Splits for Imbalance): 17.65s latency
* Prompt 5 (Outlier Capping in OLS): 24.90s latency
* **Average Latency:** 21.31s across all 5 benchmark queries.

## Repository Contents
* `app.py`: Full Streamlit web application.
* `requirements.txt`: Python package requirements.
* `prompt_test_results.json`: Raw benchmark outputs and latency metrics.
* `screenshots/`: High resolution screenshots of the application overview and verified completed GPU inference.
* `Rhydham_Shukla_Minor_Project_5_Report.pdf`: One page summary report.

## How to Run
1. Install and start Ollama:
   ollama serve
2. Pull the model:
   ollama pull deepseek-r1:7b
3. Install dependencies:
   pip install -r requirements.txt
4. Launch the application:
   python -m streamlit run app.py
