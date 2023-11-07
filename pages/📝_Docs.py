import streamlit as st


def main()-> None:
    ####################
    # Config
    ####################
    st.set_page_config(
        page_title="Docs",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Documentation")


if __name__ == "__main__":
    main()