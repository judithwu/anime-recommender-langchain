import streamlit as st
from anime_rec_langchain import generate_response

# Page title
st.set_page_config(page_title='🦜🔗 Anime Recommendation System')
st.title('🦜🔗 Anime recommender')

with st.form('myform', clear_on_submit=True):
  # Query text
  query_text = st.text_input('Ask for an anime recommendation', placeholder = "What's an anime with lots of action?")
  submitted = st.form_submit_button("Enter")

  if submitted:
    #check that something was submitted
    if query_text.strip():
      response = generate_response(query_text)
      st.info(f"{response['answer']}\nMore info: {response['sources']}")
    else:
      st.info("Please submit a question")