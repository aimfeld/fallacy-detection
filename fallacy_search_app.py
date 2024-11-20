import streamlit as st
from dotenv import load_dotenv
from src.search import fallacy_search, FallacyResponse

# Load environment variables from .env file
load_dotenv()

MAX_INPUT_LENGTH = 30000

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'response' not in st.session_state:
        st.session_state.response = None


def show_about_page():
    """Display the About page content."""
    st.header("About Fallacy Search")
    st.write("""
    This is a dummy about page for the Fallacy Search application. 

    The application is designed to help users identify potential logical fallacies in their text. 
    Simply enter your text in the main page, and our system will analyze it for common logical fallacies.
    """)


def show_main_page():
    """Display the main Fallacy Search page content."""
    st.header("Fallacy Search")

    # Instruction text
    st.write("Enter text to analyze it for logical fallacies:")

    # Text input area with character limit
    user_text = st.text_area(
        label="no label",
        label_visibility='hidden', # Hide label
        max_chars=MAX_INPUT_LENGTH,
        height=200,
        key="user_input"
    )
    user_input = user_text.strip()

    if not st.session_state.processing:
        analyze_button = st.button("Analyze", type="primary")
        if analyze_button and user_input:
            st.session_state.response = None
            st.session_state.processing = True
            st.rerun()

    if st.session_state.processing:
        with st.spinner('Processing...'):
            response = fallacy_search(user_input, model = 'gpt-4o-mini-2024-07-18')
            st.session_state.response = response
            st.session_state.processing = False
            st.rerun()

    if not st.session_state.processing and st.session_state.response:
        response: FallacyResponse = st.session_state.response
        for fallacy in response.fallacies:
            st.write(f"Fallacy: {fallacy.fallacy}")
            st.write(f"Definition: {fallacy.definition}")
            st.write(f"Span: {fallacy.span}")
            st.write(f"Reason: {fallacy.reason}")
            if fallacy.defense:
                st.write(f"Defense: {fallacy.defense}")
            st.write(f"Confidence: {fallacy.confidence}")
            st.write("")
        st.write(f"Summary: {response.summary}")
        if response.rating:
            st.write(f"Rating: {response.rating}")


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Fallacy Search",
        page_icon="🔍",
        layout="wide"
    )

    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    initialize_session_state()

    tab1, tab2 = st.tabs(["Fallacy Search", "About"])

    with tab1:
        show_main_page()
    with tab2:
        show_about_page()


if __name__ == "__main__":
    main()