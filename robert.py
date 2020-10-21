import streamlit as st
from bot import *

title = """
        <h1><a href="https://github.com/Aakash-kaushik/robert_bot">Robert Bot 🤖</a></h1>
        """

st.write("\n")
st.write("\n")

st.markdown(title, unsafe_allow_html=True)
user_input = st.text_input("Enter your Message here","Hey")
bot_output_list = eval_input(encoder, decoder, searcher, voc, user_input)
if bot_output_list != -1:
  bot_output_str=""
  for bot_output_word in bot_output_list:
      bot_output_str += bot_output_word
      bot_output_str += " "
  st.write("Robert: ", bot_output_str)
else:
  st.write("Robert: ", "Try something else human.🧍")

st.write("\n")
st.write("\n")

html_string = """
  <h2> Meet the creator </h2>
  <p align="center">
  <a href="mailto:kaushikaakash7539@gmail.com?subject = Hello from your GitHub README&body = Message"><img src="https://www.iconfinder.com/data/icons/social-icons-circular-color/512/gmail-128.png" height="60px" width="60px" alt="Gmail" ></a>
  <a href="https://www.linkedin.com/in/kaushikaakash7539/"><img src="https://www.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-linkedin-circle-128.png" height="60px" width="60px" alt="LinkedIn"></a>
  <a href="https://github.com/Aakash-kaushik"><img src="https://github.com/fluidicon.png" height="60px" width="60px alt="GitHub"></a>
  <a href="https://open.spotify.com/user/nu45gm4u9aahlsxhzt2vpige5?si=NpVR2X_rQlKyYlRLk9bdgA"><img src="https://www.iconfinder.com/data/icons/social-icons-33/128/Spotify-128.png" height="60px" width="60px alt="Sourcerer"></a> 
</p>
"""
st.markdown(html_string, unsafe_allow_html=True)

st.write("\n")
st.write("\n")

st.write("Robert Bot V2 is in the making !")
