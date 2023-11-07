from core import TFIDFSearch, BM25Search, NeuralSearch, HybridSearch


if __name__ == "__main__":
    tf_search = TFIDFSearch()
    user_input = "walmart"
    search_results = tf_search.search(user_input)

    print()
