import os
import streamlit as st
import anthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
RETRIEVAL_K = 8
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are an assistant specialized in J.K. Rowling's Harry Potter books.
You answer questions based EXCLUSIVELY on the passages provided in each message. Each passage is labeled with its source file.

Strict rules:
- You NEVER use your general knowledge or training memory about Harry Potter.
- If the provided passages contain information that answers the question — directly or by clear implication — use it to answer and quote the relevant passage.
- Only if the passages truly contain no relevant information, reply exactly: "I cannot find this information in the available passages."
- You never invent facts not supported by the provided passages.
- Always reply in the same language as the user's question.
- At the very end of your answer, on a new line, write exactly: SOURCES_USED: followed by the source filenames you actually used, comma-separated."""

HYDE_PROMPT = """Write a short passage (3-5 sentences) in the style of J.K. Rowling's Harry Potter books that would directly answer the following question. Write it as if it were an excerpt from the actual book — use narrative prose, character names, and the tone of the original text. Output only the passage, nothing else."""

st.set_page_config(
    page_title="Expert Harry Potter",
    page_icon="⚡",
    layout="centered",
)

st.markdown("""
<style>
    .stTextInput input { border-radius: 20px; padding: 12px; }
    .stButton button {
        background-color: #740001;
        color: white;
        border-radius: 20px;
        padding: 8px 20px;
    }
    .stButton button:hover { background-color: #5a0000; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_db() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local("harry_potter_hf_index", embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def generate_hypothetical_doc(question: str) -> str:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=HYDE_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()


def retrieve_context(question: str) -> tuple[str, list]:
    db = load_vector_db()
    hypothetical = generate_hypothetical_doc(question)
    seen, docs = set(), []
    for q, k in [(question, RETRIEVAL_K), (hypothetical, 35)]:
        for doc in db.similarity_search(q, k=k):
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                docs.append(doc)
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', '?')}]\n{doc.page_content}"
        for doc in docs
    )
    return context, docs


def answer_question(question: str, history: list[dict]) -> dict:
    context, docs = retrieve_context(question)

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    messages.append({
        "role": "user",
        "content": f"Passages from the Harry Potter books:\n{context}\n\nQuestion: {question}",
    })

    client = get_client()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        raw = response.content[0].text
        # Parse the SOURCES_USED line the LLM appends, then strip it from the answer
        sources, answer = [], raw
        if "SOURCES_USED:" in raw:
            body, _, sources_line = raw.rpartition("SOURCES_USED:")
            answer = body.strip()
            sources = [s.strip() for s in sources_line.split(",") if s.strip()]
        if not sources:
            sources = list({doc.metadata.get("source", "?") for doc in docs})
        return {"answer": answer, "sources": sources}

    except anthropic.AuthenticationError:
        return {
            "answer": "❌ Invalid API key. Check `ANTHROPIC_API_KEY` in your `.env` file.",
            "sources": [],
        }
    except anthropic.RateLimitError:
        return {
            "answer": "⏳ Rate limit reached. Please try again in a few seconds.",
            "sources": [],
        }
    except anthropic.APIError as e:
        return {"answer": f"❌ API error: {e.message}", "sources": []}


# ── Interface utilisateur ────────────────────────────────────────────────────

st.title("⚡ Expert Harry Potter")
st.caption("Ask me anything about the Harry Potter books!")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Welcome! I am ready to answer all your questions about the Harry Potter universe. 🧙",
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your question about Harry Potter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🔍 Searching the archives..."):
        history = st.session_state.messages[1:-1]
        response = answer_question(prompt, history)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"],
    })

    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        if response["sources"]:
            with st.expander("📚 Sources"):
                for src in response["sources"]:
                    st.write(f"• {src}")
