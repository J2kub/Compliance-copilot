import os
from pathlib import Path

import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

INDEX_DIR = Path("index") / "faiss_index"


def load_vectorstore():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nie je nastavený.")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return vs


def answer_question(question: str, vs, mode: str):
    # filter podľa režimu
    search_kwargs = {}
    if mode == "NIS2":
        search_kwargs = {"filter": {"source": "NIS2"}}
    elif mode == "GDPR":
        search_kwargs = {"filter": {"source": "GDPR"}}

    if search_kwargs:
        docs = vs.similarity_search(question, k=3, **search_kwargs)
    else:
        docs = vs.similarity_search(question, k=3)

    if not docs:
        full_answer = (
            "Z dostupnej znalostnej bázy neviem odpovedať na túto otázku.\n\n"
            "---\nZdroj: (žiadny relevantný dokument)\nTagy: Bez tagu"
        )
        return full_answer, []

    context_parts = []
    citations = []
    tag_set = set()
    used_docs = []

    for d in docs:
        src = d.metadata.get("source", "UNKNOWN")   # GDPR / NIS2
        cid = d.metadata.get("chunk_id", "1")       # napr. GDPR_1, NIS2_1
        ident = cid                                 # použijeme priamo GDPR_1 / NIS2_1

        context_parts.append(f"[{ident}] {d.page_content}")
        citations.append(f"{src} ({ident})")
        tag_set.add(src)

        used_docs.append(
            {
                "id": ident,
                "source": src,
                "content": d.page_content,
            }
        )

    context = "\n\n".join(context_parts)
    tags_str = ", ".join(sorted(tag_set)) if tag_set else "Bez tagu"

    system_prompt = (
        "Si asistent pre NIS2 a GDPR. Odpovedaj stručne (max 5 viet), "
        "používaj jednoduchú češtinu alebo slovenčinu a vychádzaj LEN z poskytnutého kontextu. "
        "V odpovedi sa pri tvrdeniach odvolávaj na zdroje v tvare GDPR (GDPR_1) alebo NIS2 (NIS2_1) "
        "podľa identifikátorov v hranatých zátvorkách v kontexte. "
        'Ak niečo v kontexte nie je, povedz napríklad: "Z dostupných informácií to neviem povedať."'
    )

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Otázka: {question}\n\nKONTEKST (bloky s identifikátormi):\n{context}",
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    cit_str = "; ".join(citations)
    full_answer = (
        f"{answer}\n\n"
        f"---\n"
        f"Zdroj: {cit_str}\n"
        f"Tagy: {tags_str}"
    )
    return full_answer, used_docs


def inject_css():
    st.markdown(
        """
        <style>
        /* len pozadie na stApp – bezpečný gradient */
        .stApp {
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(135deg, #020617, #0f172a, #0b3b3b);
            background-size: 300% 300%;
            animation: gradientMove 22s ease infinite;
        }

        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .main-block {
            max-width: 900px;
            margin: auto;
            padding: 1.2rem 1.4rem;
            background: rgba(15, 23, 42, 0.96);
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.85);
        }

        .copyright-badge {
            position: fixed;
            bottom: 16px;
            right: 16px;
            padding: 6px 14px;
            border-radius: 999px;
            background-color: rgba(15, 118, 110, 0.92);
            color: #e0f2f1;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: box-shadow 0.2s ease, transform 0.2s ease, background-color 0.2s ease;
            z-index: 999;
        }

        .copyright-badge:hover {
            box-shadow: 0 0 16px rgba(45, 212, 191, 0.95);
            transform: translateY(-1px);
            background-color: rgba(20, 184, 166, 0.98);
        }

        .mode-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.2);
            border: 1px solid rgba(45, 212, 191, 0.4);
            color: #a5f3fc;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .mode-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.8);
        }

        .streamlit-expanderHeader {
            font-size: 13px;
        }
        </style>

        <div class="copyright-badge">
            © Jakub Glončák
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="NIS2 / GDPR copilot",
        page_icon="✅",
        layout="wide",
    )

    inject_css()

    if "vs" not in st.session_state:
        with st.spinner("Načítavam index..."):
            st.session_state.vs = load_vectorstore()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    # SIDEBAR – režimy, nastavenia
    with st.sidebar:
        st.header("⚙️ Nastavenia")
        mode = st.radio(
            "Režim znalostnej bázy:",
            ["Oboje", "NIS2", "GDPR"],
            index=0,
        )
        st.caption(
            "• Oboje: kombinuje NIS2 aj GDPR\n"
            "• NIS2: len smernica NIS2\n"
            "• GDPR: len GDPR pasáže"
        )

        if st.button("♻️ Vyčistiť históriu chatu"):
            st.session_state.messages = []
            st.session_state.last_sources = []

    # Hlavný blok v strede
    with st.container():
        st.markdown('<div class="main-block">', unsafe_allow_html=True)

        st.markdown("### NIS2 / GDPR copilot")
        st.write(
            "Pýtaj sa otázky k NIS2 a GDPR. Odpovede sú generované len z malej testovacej "
            "znalostnej bázy, ktorú môžeš postupne rozširovať."
        )

        # indikátor režimu
        st.markdown(
            f"""
            <div class="mode-pill">
                <div class="mode-dot"></div>
                <span>Aktívny režim: {mode}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # História – zobrazujeme po dvojiciach (user, assistant) v expanderi
        for i in range(0, len(st.session_state.messages), 2):
            try:
                user_msg = st.session_state.messages[i]
                assistant_msg = st.session_state.messages[i + 1]
            except IndexError:
                break

            short_q = user_msg["content"].strip().replace("\n", " ")
            if len(short_q) > 60:
                short_q = short_q[:60] + "..."

            title = f"Otázka: {short_q}"
            with st.expander(title):
                with st.chat_message("user"):
                    st.markdown(user_msg["content"])
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg["content"])

        # Chat input dole
        user_input = st.chat_input("Napíš otázku k NIS2/GDPR...")

        if user_input:
            # zobraz používateľa
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # odpoveď
            with st.chat_message("assistant"):
                with st.spinner("Premýšľam..."):
                    try:
                        answer, used_docs = answer_question(
                            user_input,
                            st.session_state.vs,
                            mode,
                        )
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.session_state.last_sources = used_docs
                    except Exception as e:
                        st.error(f"Chyba: {e}")

        # blok s použitými časťami (klikateľné anchory)
        if st.session_state.last_sources:
            st.markdown("---")
            st.markdown("**Použité časti z databázy:**")
            for src in st.session_state.last_sources:
                with st.expander(f"{src['source']} ({src['id']})"):
                    st.markdown(src["content"])

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
