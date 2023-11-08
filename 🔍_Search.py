import streamlit as st
from core import (
    TFIDFSearch,
    BM25Search,
    NeuralSearch,
    HybridSearch,
    RETRIVAL_MODELS,
    MEAN_TYPES,
    NORM_TYPES,
    SCORE_TYPES,
)


def main():
    """App runner"""

    ####################
    # Config
    ####################
    st.set_page_config(
        page_title="Search",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ####################
    # Options
    ####################
    with st.sidebar:
        # st.title("Options")

        search_modes = ["TF-IDF", "BM25", "Neural", ":rainbow[**Hybrid**] :zap:"]

        search_mode_container = st.container()
        with search_mode_container:
            st.header("Search Mode", divider="blue")
            s_mode = st.radio(
                "Search Mode",
                search_modes,
                label_visibility="collapsed",
                horizontal=False,
            )
            with st.container():
                mean_type = None
                normalize = None
                score_type = None
                norm_type = None
                match s_mode:
                    case "Neural":
                        st.header("Mode Settings", divider="blue")
                        st.subheader("Model")
                        model = st.radio(
                            "Model",
                            RETRIVAL_MODELS,
                            label_visibility="collapsed",
                            horizontal=False,
                        )
                        st.subheader("Score Type")
                        score_type = st.radio(
                            "Score Type",
                            SCORE_TYPES,
                            label_visibility="collapsed",
                            horizontal=False,
                        )
                    case ":rainbow[**Hybrid**] :zap:":
                        st.header("Mode Settings", divider="blue")
                        st.subheader("Neural Model")
                        model = st.radio(
                            "Model",
                            RETRIVAL_MODELS,
                            label_visibility="collapsed",
                            horizontal=False,
                        )
                        st.subheader("Combine Strategy")
                        mean_type = st.radio(
                            "Mean Type",
                            MEAN_TYPES,
                            label_visibility="collapsed",
                            horizontal=False,
                        )
                        st.subheader("Normalization")
                        normalize = st.toggle("Normalize", value=True)
                        if normalize:
                            norm_type = st.radio(
                                "Norm Type",
                                NORM_TYPES,
                                label_visibility="collapsed",
                                horizontal=False,
                            )
                        st.subheader("Score Type")
                        score_type = st.radio(
                            "Score Type",
                            SCORE_TYPES,
                            label_visibility="collapsed",
                            horizontal=False,
                        )

        filter_strategy_container = st.container()
        with filter_strategy_container:
            st.header("Filter Startegy", divider="blue")
            # f_strat = st.radio("Filter Startegy", filter_modes, label_visibility="collapsed", horizontal=True)

            strategy_inputs = st.container()
            with strategy_inputs:
                top_k_check = st.checkbox("Top-K", value=True)
                top_k = 20
                if top_k_check:
                    top_k = st.slider(
                        "Value",
                        min_value=1,
                        max_value=100,
                        value=20,
                        step=1,
                        label_visibility="collapsed",
                    )

                threshold_check = st.checkbox("Threshold")
                thresh = 0.05
                if threshold_check:
                    thresh = st.slider(
                        "Value",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.55,
                        step=0.01,
                        label_visibility="collapsed",
                    )

                cluster_check = st.checkbox("Cluster")
                min_dis = 0.35
                if cluster_check:
                    min_dis = st.slider(
                        "Minimum cluster distance",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.35,
                        step=0.01,
                        label_visibility="visible",
                    )

    ####################
    # Main Content
    ####################

    st.markdown('<div align="center"><div style="font-size:3rem; font-weight:600">Offer Search App</div><div color="#3d9df3" style="border-radius: 3px; border:none; background-color: rgb(61, 157, 243); width:100%; margin-top: 0.5rem; margin-bottom: 2rem; font-size:0.1rem; color:background-color: rgb(61, 157, 243);">.</div></div>',unsafe_allow_html=True)

    query = st.text_input(
        "Search", placeholder="Type to search offers", label_visibility="collapsed"
    )

    with st.spinner("Loading..."):
        match s_mode:
            case "TF-IDF":
                searcher = TFIDFSearch()
            case "BM25":
                searcher = BM25Search()
            case "Neural":
                searcher = NeuralSearch(model=model, score_type=score_type)
            case ":rainbow[**Hybrid**] :zap:":
                searcher = HybridSearch(model=model, score_type=score_type)

        placeholder = st.empty()
        placeholder.success("", icon="✔")
        placeholder.empty()

    if query.strip():
        with st.spinner(""):
            colun_config = {
                "index": None,
                "SCORE": "Score",
                "OFFER": "Offer",
                "RETAILER": "Retailer",
                "BRAND": "Brand",
                "CATEGORIES": "Categories",
                "SUPER_CATEGORIES": None,
                "CLUSTER": None,
            }

            result = searcher.search(
                query=query.strip(),
                mean_type=mean_type,
                norm_type=norm_type,
                top_k=top_k,
                threshold=thresh,
                dis_threshold=min_dis,
                e_top_k=top_k_check,
                e_threshold=threshold_check,
                e_cluster=cluster_check,
            )
            if result.shape[0] > 0:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                    height=600,
                    column_config=colun_config,
                )
                st.success(f"Found {result.shape[0]} results !")
            else:
                st.error("No resuls found")

if __name__ == "__main__":
    main()
