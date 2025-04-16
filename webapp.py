import streamlit as st

st.title("Yelp Recommender")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Enter something",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "User", "content": prompt})

    response = f"Echo: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "Yelp", "content": response})
