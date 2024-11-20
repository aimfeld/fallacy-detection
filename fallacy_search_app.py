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
    if 'error' not in st.session_state:
        st.session_state.error = None


def show_about_page():
    """Display the About page content."""
    st.header("About Fallacy Search")
    st.write("""
    The Fallacy Search application is designed to help users identify potential logical fallacies in any text. 
    It uses GPT-4o to detect and analyze common reasoning errors and argumentative flaws.
    The tool aims to promote critical thinking and improve the quality of public discourse by making logical analysis 
    more accessible.
    
    Fallacy Search has been developed and benchmark tested by Adrian Imfeld as part of a data science master thesis at 
    Lucerne University of Applied Sciences and Arts (HSLU). For more information, see GitHub link below.
    """)

    st.write("Potential Applications:")
    st.markdown("""
    - Journalism and fact-checking
    - Highlighting manipulative content and propaganda  
    - Analyzing blog posts and content on social media
    - Identifying weaknesses in legal arguments and court decisions
    - Flagging pseudoscience and unsound marketing
    """)
    st.write('👤 Author: Adrian Imfeld')
    st.write('📧 E-Mail: aimfeld@aimfeld.ch')
    st.write('🔗 GitHub: [aimfeld/fallacy-detection](https://github.com/aimfeld/fallacy-detection)')

def get_star_rating(rating: int):
    return '⭐' * rating + '☆' * (10 - rating)


def show_main_page():
    """Display the main Fallacy Search page content."""
    st.header("Fallacy Search")

    # Instruction text
    st.write("Enter text to detect logical fallacies and get a reasoning score:")

    # Text input area with character limit
    user_text = st.text_area(
        label="no label",
        label_visibility='hidden', # Hide label
        max_chars=MAX_INPUT_LENGTH,
        height=100,
        key="user_input"
    )
    user_input = user_text.strip()

    if not st.session_state.processing:
        analyze_button = st.button("Analyze", type="primary")
        if analyze_button and user_input:
            st.session_state.response = None
            st.session_state.error = None
            st.session_state.processing = True
            st.rerun()

    if st.session_state.error:
        st.error(f"An error occurred: {st.session_state.error}")
        st.write("Please try again.")

    if st.session_state.processing:
        with st.spinner('Processing...'):
            try:
                # response = fallacy_search(user_input, model = 'gpt-4o-mini-2024-07-18')
                response = fallacy_search(user_input)
                st.session_state.response = response
            except Exception as e:
                st.session_state.error = e

        st.session_state.processing = False
        st.rerun()

    if not st.session_state.processing and st.session_state.response:
        response: FallacyResponse = st.session_state.response
        # Create a container for better spacing
        with st.container():
            if response.fallacies:
                st.markdown("### Detected Fallacies")
                # Display each fallacy in an expander
                for i, fallacy in enumerate(response.fallacies, 1):
                    with st.expander(f'🎯 **Fallacy #{i}: {fallacy.fallacy}**', expanded=True):
                        st.markdown(f'**{fallacy.fallacy}:** {fallacy.definition}')
                        st.markdown(f'**Quote:** _“{fallacy.span}”_')
                        st.markdown(f'**Reason:** {fallacy.reason}')
                        if fallacy.defense:
                            st.markdown(f'**Defense:** {fallacy.defense}')
                        st.markdown(f'**Confidence:** {fallacy.confidence * 100:.0f}%')

                # Display summary and rating in a separate container
                st.markdown("---")

            st.markdown("### Overall Analysis")
            st.markdown("**📝 Summary**")
            st.markdown(response.summary)

            st.markdown(f'**⭐ Reasoning Score**')
            if response.rating:
                st.markdown(f'{get_star_rating(response.rating)} ({response.rating} out of 10)')
            else:
                st.markdown("Not rated since the text seems to contain no arguments.")


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