# Analysis Report
---

## Project Overview
The goal of this project was to build a tool that enables users to intelligently search for offers using text input. The tool is designed to enhance user experience within the Fetch app by providing relevant offers based on the user's search queries.

I approached the project with a strong emphasis on ensuring ease of deployment and scalability as key priorities. In this analysis report, I will detail the approach taken to address the problem, the data used, the preprocessing steps, the models and similarity scoring methods employed, the tool's deployment, and the results achieved.

## Problem Statement

The primary objective of this project is to develop a text-based search tool that allows users to intelligently search for offers within the Fetch app. Users should be able to input text queries related to product categories, specific brands, or retailer names, and the tool should return a list of relevant offers, along with a score indicating the similarity between the user's input and each offer. The problem can be broken down into the following key components:

1. **Category Search:** When a user searches for a product category (e.g., "diapers"), the tool should retrieve a list of offers that are relevant to that specific category.

2. **Brand Search:** If a user searches for a particular brand (e.g., "Huggies"), the tool should provide a list of offers associated with that brand.

3. **Retailer Search:** Users may also search for offers available at a particular retailer (e.g., "Target"). The tool should return a list of offers that are relevant to that specific retailer.

4. **Similarity Scoring:** In addition to returning relevant offers, the tool should assign a score to each offer to measure the similarity between the user's input and the offer text.

The success of this project is contingent on the tool's ability to accurately identify and retrieve relevant offers, brands, and retailers based on user text input. Additionally, the similarity scores provided should assist users in quickly assessing the relevance of each returned offer.


## Approach

In this section, I will outline my approach to solving the problem of enabling users to intelligently search for offers via text input, considering categories, brands, and retailers.

### Data

For this task, I had access to three essential datasets:

1. **brand_category.csv**: This dataset consists of a mapping of brands to their respective categories, along with the number of receipts associated with each brand and category. In total, we were provided with 9,907 brand-category relationships.

2. **categories.csv**: This dataset provides information about categories and their relationships to super categories. There were 119 category-super category relationships provided.

3. **offer_retailer.csv**: This dataset contains information about various offers, including details about the associated brands and retailers. In total, we had access to 385 offers in this dataset.

### Model and Algorithm

The tool provide three search types:
1. **Keyword Based Search**: <Write about keyword based search>
    Types Employed:
    1. TF-IDF: <write about TFIDF>
    2. BM25: <write about BM25>

1. **Symantic Search** <Write about symantic based search>
    Models Used:
    1. TF-IDF: <write about TFIDF>
    2. BM25: <write about BM25>
1. **Hybrid Search**
I employed a Natural Language Processing (NLP) approach to match user queries with relevant offers. Specifically, we used a text embedding technique to represent offers and user input as numerical vectors. We considered the following approaches:

- **Word Embeddings**: We used pre-trained word embeddings (e.g., Word2Vec, FastText) to convert text data into dense vector representations. These embeddings capture semantic relationships between words.

- **Sentence Embeddings**: We experimented with techniques like Doc2Vec and Universal Sentence Encoder to convert entire offers and user queries into embeddings, which allows for similarity scoring.

### Data Preprocessing

Data preprocessing was crucial for cleaning and preparing the datasets for analysis. Key data preprocessing steps included:

- **Preprocessing**: I performed data preprocessing and merged the 3 datasets provided in to one main dataset. This dataset has the Offer, Retailer, Brand, Categories, Super Categories and a Target for modeling. This dataset was stored in a sqlite database for easy and fast retrival. To scale the system a better alternative then SQLite such as MySQL can be used just by replacing the url in the code.
- **Text Preprocessing**: Different type of text processing was applied to We performed text cleaning, lowercasing, and tokenization to standardize text inputs and improve matching accuracy.
- **Handling Missing Data**: We addressed any missing or incomplete data in the datasets. For example, some offers might have missing retailer or brand information.
- **Data Integration**: We combined information from the two datasets to create a unified dataset for efficient searching and matching.


### Similarity Scoring

To measure the similarity between user input and offers, we employed cosine similarity as our similarity metric. This metric helps us quantify how similar two vectors are, indicating the relevance of an offer to the user's query. Offers were ranked based on their similarity scores, and the top matches were presented to the user.

### Tool Deployment

For deployment, we developed a web-based tool that allows users to input text queries and receive relevant offers. The tool provides a user-friendly interface where users can search for categories, brands, and retailers, and it displays the most relevant offers with their associated similarity scores.

The tool was deployed on a web server, and it can be accessed via the provided link for user evaluation and feedback.

Our approach leverages NLP techniques and similarity scoring to provide users with a streamlined and intelligent search experience, meeting the acceptance criteria set by the project.



## Results
Present the results of your analysis, including performance metrics and examples of successful searches. If you have any visualizations or tables, include them here.

## Discussion
Discuss the strengths and weaknesses of your approach. Mention any challenges you encountered during the project and how you overcame them. Discuss the trade-offs you considered when designing the tool.

## Conclusion
Summarize the key findings and the impact of your work. Discuss how well the tool meets the acceptance criteria.

## Future Work
Suggest possible improvements or future work that could enhance the tool's capabilities or address limitations.

## Repository and Deployment Links
Provide links to your GitHub repository containing the code and any hosted version of your tool (if applicable).

## Running the Tool Locally (if applicable)
If your tool can be run locally, provide instructions on how to set it up and use it. Include any dependencies and configurations required.

## Acknowledgments
If you used external libraries, datasets, or other resources, acknowledge them here.

## References
List any references, research papers, or articles that you found useful during the project.