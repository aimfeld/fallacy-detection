import streamlit as st
from dotenv import load_dotenv
from src.search import get_search_system_prompt, fallacy_search, get_fallacy_response_string

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

    Note: This is a demonstration version with placeholder analysis results.
    """)


def show_main_page():
    """Display the main Fallacy Search page content."""
    st.header("Fallacy Search")

    # Instruction text
    st.write("Some instructions here.")

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
        if analyze_button:
            st.session_state.response = None
            st.session_state.processing = True

    if user_input and st.session_state.processing:
        with st.spinner('Processing...'):
            response = fallacy_search(user_input, model = 'gpt-4o-mini-2024-07-18')
            st.session_state.response = response
            st.session_state.processing = False

    if st.session_state.response:
        st.write(get_fallacy_response_string(st.session_state.response))


def main():
    """Main application function."""
    # Set light mode and page config
    st.set_page_config(
        page_title="Fallacy Search",
        page_icon="🔍",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Create tabs
    tab1, tab2 = st.tabs(["Fallacy Search", "About"])

    # Display content based on selected tab
    with tab1:
        show_main_page()
    with tab2:
        show_about_page()


if __name__ == "__main__":
    main()