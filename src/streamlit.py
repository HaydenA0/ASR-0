import streamlit as st
from globals import PROJECT_SOURCE
 
st.sidebar.title("Audio Recording App")
st.title("Record Your Audio")
st.write("Press the button to start recording and then stop when you're done.")
audio = st.audio_input("Record your audio")
 
if audio:
    with open(PROJECT_SOURCE + "/audio/buffer.wav", "wb") as f:
        f.write(audio.getbuffer())
        st.write("Audio recorded and saved successfully!")

