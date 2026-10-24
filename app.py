"""
========================================================================================
UNLOX ACADEMY - MACHINE LEARNING AUGUST 2026 BATCH
MINOR PROJECT 5: OLLAMA + STREAMLIT INTERACTIVE AI APPLICATION
========================================================================================
Student Name : Rhydham Shukla
Batch        : ML (Machine Learning) 2026 August Batch
Institution  : Unlox Academy
Application  : OmniStudy - AI Academic Study Assistant & Concept Explainer
Backend LLM  : Ollama (deepseek-r1:7b / lady-testarossa:latest on NVIDIA RTX 5050 GPU)
Frontend     : Streamlit 1.66
========================================================================================
"""

import time
import re
import requests
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="OmniStudy | AI Academic Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Quality CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0B2545;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #134074;
        margin-bottom: 1.2rem;
    }
    .student-badge {
        background: linear-gradient(135deg, #EEF4F8 0%, #D0D7DE 100%);
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 4px solid #134074;
        font-size: 0.85rem;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


def check_ollama_health(host: str = "http://127.0.0.1:11434") -> tuple[bool, list[str]]:
    """Checks if the local Ollama server is running and retrieves installed models."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []


def query_ollama(
    prompt: str,
    model: str = "deepseek-r1:7b",
    system_prompt: str = "",
    temperature: float = 0.7,
    host: str = "http://127.0.0.1:11434"
) -> dict:
    """Sends prompt to local Ollama REST endpoint and returns generated response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }
    start_time = time.time()
    try:
        response = requests.post(f"{host}/api/generate", json=payload, timeout=90.0)
        latency = time.time() - start_time
        if response.status_code == 200:
            res_json = response.json()
            return {
                "success": True,
                "response": res_json.get("response", ""),
                "latency_s": round(latency, 2),
                "eval_count": res_json.get("eval_count", 0),
                "eval_duration": res_json.get("eval_duration", 0)
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}", "latency_s": round(latency, 2)}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (>90s). The model may still be loading into GPU VRAM.", "latency_s": round(time.time() - start_time, 2)}
    except Exception as e:
        return {"success": False, "error": f"Connection failed: {str(e)}", "latency_s": round(time.time() - start_time, 2)}


def parse_reasoning_and_answer(raw_text: str) -> tuple[str, str]:
    """Separates <think> tags from final response for clean UI display."""
    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return reasoning, answer
    return "", raw_text.strip()


def main():
    # Sidebar: Student Credentials & Settings
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.markdown("### 🎓 Academic Submission")
        st.markdown("""
        <div class="student-badge">
            <b>Student:</b> Rhydham Shukla<br/>
            <b>Batch:</b> ML August 2026 Batch<br/>
            <b>Institution:</b> Unlox Academy<br/>
            <b>Project:</b> Minor Project 5 (Ollama + Streamlit)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Engine Configuration")

        is_healthy, available_models = check_ollama_health()
        if is_healthy:
            st.success("🟢 Local Ollama Daemon Online", icon="✅")
            model_options = available_models if available_models else ["deepseek-r1:7b"]
            default_idx = model_options.index("deepseek-r1:7b") if "deepseek-r1:7b" in model_options else 0
            selected_model = st.selectbox("Active Local Model", model_options, index=default_idx)
        else:
            st.error("🔴 Ollama Offline (Run `ollama serve`)", icon="⚠️")
            selected_model = "deepseek-r1:7b"

        temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

        st.markdown("---")
        st.markdown("### 📚 Quick Curriculum Prompts")
        sample_prompts = [
            "Explain Overfitting vs Underfitting and 3 ways to fix it.",
            "Why do we use the Sigmoid function in Logistic Regression?",
            "Compare Random Forest vs Decision Trees.",
            "Explain stratified train-test split for imbalanced datasets.",
            "Why cap 95th percentile outliers in NYC Airbnb regression?"
        ]
        for p in sample_prompts:
            if st.button(f"👉 {p[:32]}...", use_container_width=True):
                st.session_state["queued_prompt"] = p

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main Chat View
    st.markdown('<div class="main-title">OmniStudy: AI Academic Study Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Powered by Local Ollama Large Language Model & Streamlit | Unlox ML 2026 August Batch</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello Rhydham! I am your **OmniStudy AI Assistant**, running locally via Ollama on your system. How can I help you master Machine Learning concepts today?"
            }
        ]

    # Display prior conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "reasoning" in msg and msg["reasoning"]:
                with st.expander("🧠 Internal Model Reasoning (CoT)"):
                    st.markdown(msg["reasoning"])
            st.markdown(msg["content"])
            if "latency" in msg:
                st.caption(f"⚡ Inferred locally in {msg['latency']}s")

    # Handle incoming prompt
    prompt_to_process = None
    if "queued_prompt" in st.session_state and st.session_state["queued_prompt"]:
        prompt_to_process = st.session_state["queued_prompt"]
        st.session_state["queued_prompt"] = None
    else:
        user_input = st.chat_input("Ask any Machine Learning question or paste code here...")
        if user_input:
            prompt_to_process = user_input

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Assistant generation
        with st.chat_message("assistant"):
            with st.spinner(f"OmniStudy is synthesizing response via {selected_model}..."):
                system_prompt = (
                    "You are OmniStudy, an elite Machine Learning tutor for student Rhydham Shukla "
                    "at Unlox Academy. Provide mathematically precise, structured, and pedagogical explanations. "
                    "Use KaTeX math formulas and bullet points where helpful."
                )
                res = query_ollama(
                    prompt=prompt_to_process,
                    model=selected_model,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

                if res.get("success"):
                    raw_text = res["response"]
                    reasoning, clean_answer = parse_reasoning_and_answer(raw_text)

                    if reasoning:
                        with st.expander("🧠 Internal Model Reasoning (CoT)"):
                            st.markdown(reasoning)

                    st.markdown(clean_answer)
                    st.caption(f"⚡ Inferred locally on GPU in {res['latency_s']}s")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_answer,
                        "reasoning": reasoning,
                        "latency": res['latency_s']
                    })
                else:
                    st.error(f"Error querying model: {res.get('error')}")


if __name__ == "__main__":
    main()
