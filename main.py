import streamlit as st
import torch
import wikipedia
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

def is_repetitive(text):
    words = text.split()
    return len(words) != len(set(words))

def low_confidence(response):
    if len(response.split()) < 5:
        return True
    if is_repetitive(response):
        return True
    if "i don't know" in response.lower():
        return True
    return False

def retrieve_knowledge(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None

def generate(bot_input_ids):
    return model.generate(
        bot_input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

st.set_page_config(page_title="CRAG Chatbot", layout="centered")

st.title("CRAG Chatbot (Corrective RAG)")
st.markdown("Transformer chatbot with **Corrective RAG**")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors='pt'
    ).to(device)

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat(
            [st.session_state.chat_history_ids, new_input_ids],
            dim=-1
        )
    else:
        bot_input_ids = new_input_ids

    bot_input_ids = bot_input_ids[:, -1000:]

    st.session_state.chat_history_ids = generate(bot_input_ids)

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if low_confidence(response):
        with st.chat_message("assistant"):
            st.markdown("Let me verify that...")

        knowledge = retrieve_knowledge(user_input)

        if knowledge:
            augmented_input = f"Context: {knowledge}\nUser: {user_input}\nBot:"
            augmented_ids = tokenizer.encode(
                augmented_input,
                return_tensors='pt'
            ).to(device)

            st.session_state.chat_history_ids = generate(augmented_ids)

            response = tokenizer.decode(
                st.session_state.chat_history_ids[:, augmented_ids.shape[-1]:][0],
                skip_special_tokens=True
            )

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

st.sidebar.title("Controls")

if st.sidebar.button("Reset Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.messages = []
    st.sidebar.success("Chat reset!")
