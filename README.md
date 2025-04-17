# Home_assignment4
Name:Siva gadiraju
Number:700456448

Natural Language Processing and Attention Mechanism Demos
Table of Contents
Named Entity Recognition with spaCy

Text Preprocessing with NLTK

Scaled Dot-Product Attention (Transformer-style)

Sentiment Analysis with Hugging Face Transformers

Dependencies & Setup

1. Named Entity Recognition with spaCy
This section demonstrates how to identify and extract named entities (like people, locations, dates, and works of art) using the spaCy NLP library. The input sentence is processed by spaCy’s pre-trained English model to detect entities and classify them with labels such as PERSON, DATE, GPE (Geopolitical Entity), and more.

2. Text Preprocessing with NLTK
Here, we walk through a basic NLP preprocessing pipeline using the Natural Language Toolkit (NLTK). The steps include:

Tokenization: Breaking the sentence into individual words.

Stopword Removal: Filtering out commonly used words that carry little meaning (like “the”, “is”, etc.).

Stemming: Reducing words to their root forms using the Porter Stemmer (e.g., "running" → "run").

This pipeline helps prepare text data for downstream tasks like classification or clustering.

3. Scaled Dot-Product Attention (Transformer-style)
This section implements the core attention mechanism used in transformer architectures like BERT and GPT. It includes the following steps:

Calculating the dot product between query and key vectors.

Scaling the result by the square root of the vector dimension.

Applying the softmax function to get attention weights.

Using these weights to generate a weighted sum of value vectors.

The output demonstrates how different input vectors attend to each other in a simplified attention model.

4. Sentiment Analysis with Hugging Face Transformers
This part uses the Hugging Face transformers library to perform sentiment analysis on text input. It utilizes a pre-trained model (by default, distilbert-base-uncased-finetuned-sst-2-english) to classify text as positive or negative, along with a confidence score. This shows how transformer-based models can be applied for real-world NLP tasks with minimal code.

5. Dependencies & Setup
To run all sections of this project, the following Python packages are required:

spaCy (with the English model en_core_web_sm)

NLTK (with datasets like punkt and stopwords)

NumPy and SciPy (for matrix operations and softmax)

transformers (by Hugging Face for sentiment analysis)

All dependencies can be installed via pip, and the required language models and data corpora can be downloaded using built-in commands.
