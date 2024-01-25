import streamlit as st
from anime import generate_response

# Page title
st.set_page_config(page_title='🦜🔗 Anime Recommendation System')
st.title('🦜🔗 Anime recommender')
#st.image("./gojo.png")

# Query text
query_text = st.text_input('Ask for an anime recommendation', placeholder = "What's an anime with lots of action?")

with st.form('myform', clear_on_submit=True):
  response = generate_response(query_text)

st.info(f"{response['answer']}\nMore info: {response['sources']}")
#st.text(f"More info: {response['sources']}")