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

    with open('./pages/analysis.md', 'r', encoding='utf8') as file:
        page = file.read()
    st.markdown(page)


if __name__ == "__main__":
    main()