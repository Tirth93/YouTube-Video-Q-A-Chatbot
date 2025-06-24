import streamlit as st
from dotenv import load_dotenv
from utils import get_transcript
from qa_chain import build_qa_chain

load_dotenv()

st.set_page_config(page_title="YouTube Video Q&A Chatbot", page_icon="🎥", layout="centered")

st.title("🎥 YouTube Video Q&A Chatbot")

video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    with st.spinner("Fetching transcript and building vector store..."):
        transcript = get_transcript(video_url)
        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            qa = build_qa_chain(transcript)
            st.success("Model is ready! Ask your questions.")

            query = st.text_input("Ask a question about the video:")
            if query:
                with st.spinner("Generating answer..."):
                    response = qa.run(query)
                    st.markdown("### 💬 Answer:")
                    st.write(response)
# dummy update to push as Tirth93