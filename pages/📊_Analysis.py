import streamlit as st


def main()-> None:
    ####################
    # Config
    ####################
    st.set_page_config(
        page_title="Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Analysis")


if __name__ == "__main__":
    main()