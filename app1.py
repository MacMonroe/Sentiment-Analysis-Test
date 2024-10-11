import streamlit as st
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from collections import Counter
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import pandas as pd
from nltk import trigrams
from nltk.corpus import stopwords
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import string
from nrclex import NRCLex
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add this at the beginning of your Streamlit app
# Add this at the beginning of your Streamlit app
# custom_css = """
#     <style>
        
#         body, .stApp {
#             background-color: #E0BBE4 !important;
#         }

#         .stExpander > summary {
#             border: 2px solid purple;   /* Purple border for the expander */
#             border-radius: 4px;         /* Optional: Rounded corners */
#         }

#         div[data-baseweb="accordion"] > div {
#             border: 2px solid purple !important;
#         }

#         .element-container .stTable table {
#             background-color: #fff;
#             border-collapse: collapse;
#             width: 100%;
#             margin: 20px 0;
#             box-shadow: 0 0 10px rgba(0, 0, 0, 0.15);
#         }

#         .element-container .stTable table th {
#             background-color: #fff;
#             color: #333;  /* Dark gray column names */
#             border-bottom: 2px solid #ddd;
#             font-weight: bold;
#         }

#         .element-container .stTable table td {
#             border: 1px solid #ddd;
#             padding: 10px 15px;
#         }

#         .element-container .stTable table tr:hover {
#             background-color: #f5f5f5;
#         }

#         .stButton>button {
#             background-color: #FF1493;
#             color: white;
#             padding: 10px 15px;
#             border-radius: 5px;
#             border: none;
#             font-weight: bold;
#             transition: background-color 0.3s;
#         }

#         .stButton>button:hover {
#             background-color: #C0117D;
#         }
#     </style>
# """

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# st.markdown(custom_css, unsafe_allow_html=True)

#     <style>
#         body, .stApp {
#             background-color: #FFC0CB !important;
#         }
#     </style>
# """


# st.markdown(custom_css, unsafe_allow_html=True)

# NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')



def display_wordcloud(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # Filter words based on criteria: not a stop word, not punctuation, and at least 4 letters long
    words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation and len(word) >= 4]
    
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(30)  # Get the top 30 words

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(most_common_words))

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.close()


def top_bigrams(text, n=10):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuation and filter words based on length
    words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation and len(word) >= 4]
    
    # Remove quotes and extra spaces
    words = [word.replace('"', '').replace("'", "").strip() for word in words]
    
    bigram_list = list(bigrams(words))
    
    # We don't need additional filtering here since words already satisfy the conditions
    bigram_freq = Counter(bigram_list)
    return bigram_freq.most_common(n)

# Average Length of a Sentence
def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    return avg_len

# Average Words Between Punctuation
def avg_words_between_punctuation(text):
    splits = re.split(r'[{}]+'.format(string.punctuation), text)
    counts = [len(s.split()) for s in splits if s]
    return sum(counts) / len(counts)

def analyze_text(text):
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # 1. Sentence Analysis
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    
    short_sentences = sum(1 for s in sentences if len(s.split()) <= 10)
    medium_sentences = sum(1 for s in sentences if 11 <= len(s.split()) <= 20)
    long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
    
    # 2. Pauses Analysis
    num_commas = text.count(',')
    
    # 3. Sentiment Analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return avg_sentence_length, short_sentences, medium_sentences, long_sentences, num_commas, polarity, subjectivity

# # Top 5 Topics using LDA
# def lda_topics(text, n=5):
#     vectorizer = CountVectorizer(stop_words='english')
#     data_vectorized = vectorizer.fit_transform([text])
#     lda = LatentDirichletAllocation(n_components=n, random_state=42)
#     lda.fit(data_vectorized)
#     topics = []
#     for idx, topic in enumerate(lda.components_):
#         topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-2:][::-1]]
#         topics.append(' '.join(topic_words))
#     return topics

def bert_topics(text, n=5):
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Initialize tokenizer and model using the "auto" classes
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for sentence in sentences:
        tokens = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            output = model(**tokens)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    topics = []
    for center in cluster_centers:
        distances = [(i, torch.nn.functional.cosine_similarity(torch.tensor(center), torch.tensor(embedding), dim=0)) for i, embedding in enumerate(embeddings)]
        closest = sorted(distances, key=lambda x: x[1], reverse=True)[:2]
        topic_sentences = ' '.join(sentences[i] for i, _ in closest)
        topics.append(topic_sentences)

    return topics

def display_sentiment_over_time(text):
    sentences = nltk.sent_tokenize(text)
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]

    plt.figure(figsize=(10, 5))
    plt.plot(sentiments, color='hotpink')
    plt.ylabel('Sentiment Polarity')
    plt.xlabel('Sentence Number')
    plt.tight_layout(pad=3)  # To ensure some space between the axes and the end of the image
    st.pyplot(plt)
    plt.close()

def display_emotion_analysis(text):
    text_object = NRCLex(text)
    emotion_frequencies = text_object.affect_frequencies

    plt.figure(figsize=(10, 5))
    plt.bar(emotion_frequencies.keys(), emotion_frequencies.values(), color='hotpink')
    plt.ylabel('Frequency')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=3)  # To ensure some space between the axes and the end of the image
    st.pyplot(plt)
    plt.close()


# Average Words Between Punctuation
def avg_words_between_punctuation(text):
    splits = re.split(r'[{}]+'.format(string.punctuation), text)
    counts = [len(s.split()) for s in splits if s]
    return sum(counts) / len(counts) if counts else 0


# Total Words
def total_words(text):
    return len(nltk.word_tokenize(text))

def word_complexity(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    
    # Categorize words based on length
    complexity_counts = {
        '3 letters': sum(1 for word in words if len(word) == 3),
        '4 letters': sum(1 for word in words if len(word) == 4),
        '5 letters': sum(1 for word in words if len(word) == 5),
        '6 letters': sum(1 for word in words if len(word) == 6),
        '7 letters': sum(1 for word in words if len(word) == 7),
        '8 letters': sum(1 for word in words if len(word) == 8),
        '9 letters': sum(1 for word in words if len(word) == 9),
        '10 letters': sum(1 for word in words if len(word) == 10),
        '11 letters': sum(1 for word in words if len(word) == 11),
        '12 letters': sum(1 for word in words if len(word) == 12),
    }
    
    return complexity_counts

def top_trigrams(text, n=10):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    trigram_list = list(trigrams(words))
    trigram_freq = Counter(trigram_list)
    return trigram_freq.most_common(n)


# Streamlit App
st.markdown("# Text Analysis Tool 🫧💭")


with st.container():
    user_input = st.text_area("Enter the text for analysis:")

    if len(user_input) < 500 and user_input:
        st.warning("Please enter more text (at least 500-1000 words for a good answer) Let's go! :smile: ")

    else:
        
        main_col1,_, main_col2 = st.columns([1,3,1])

        if main_col1.button('Analyze'):

            with main_col2:
                # Display Total Words
                total = total_words(user_input)
                st.markdown("Total Words: "+ str(total))
                
                # st.write(total)

            avg_sentence_length, short_sentences, medium_sentences, long_sentences, num_commas, polarity, subjectivity = analyze_text(user_input)

            st.header("")

            col1, col2 = st.columns(2)

            with col1:

                with st.expander("Sentence Analysis") :
                # Display Sentence Analysis
                # st.markdown("## Sentence Analysis")


                    st.write(f"Average Sentence Length: Approximately {avg_sentence_length:.2f} words.")
                    st.write(f"Short Sentences (≤10 words): {short_sentences} sentences.")
                    st.write(f"Medium Sentences (11-20 words): {medium_sentences} sentences.")
                    st.write(f"Long Sentences (>20 words): {long_sentences} sentences.")
            
            with col2:
                # Display Pauses Analysis
                # st.subheader("Pauses Analysis")
                with st.expander("Pauses Analysis") :
                # st.markdown("## Pauses Analysis")
                    st.write(f"Number of Pauses (commas): {num_commas} occurrences.")

            with col1:
                with st.expander("Sentiment Analysis") :
            # Display Sentiment Analysis
            # st.subheader("Sentiment Analysis")
                    st.write(f"Polarity: ~{polarity:.2f} (Note: Polarity ranges from -1 to 1, where -1 is very negative, 1 is very positive, and 0 is neutral).")
                    st.write(f"Subjectivity: ~{subjectivity:.2f} (Note: Subjectivity ranges from 0 to 1, where 0 is very objective and 1 is very subjective).")

            # Display Average Words Between Punctuation

            with col2:
                with st.expander("Average Words Between Punctuation") :
            # st.subheader("Average Words Between Punctuation")
                    avg_punct = avg_words_between_punctuation(user_input)
                    st.write(f"After every {avg_punct:.2f} words on average, a punctuation is used.")

           # Sentiment Over Time
            st.subheader("Sentiment Over Time")
            display_sentiment_over_time(user_input)

            # Emotion Analysis
            st.subheader("Emotion Analysis")
            display_emotion_analysis(user_input)

                    

            # Word Complexity
            complexity_result = word_complexity(user_input)
            st.subheader("Word Complexity")
            complexity_df = pd.DataFrame(list(complexity_result.items()), columns=['Word Length', 'Count'])
            st.table(complexity_df)
            
        # Display Top 30 Keywords in a table
            st.subheader("Top 30 Keywords")
            keywords = display_wordcloud(user_input)
            
            if keywords:  # Check if keywords list is not empty
                keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
                st.table(keywords_df)
            else:
                st.write("No keywords found!")
            
            # Display Top 10 Bigrams in a table
            st.subheader("Top 10 Bigrams (Group of 2 words)")
            bigrams_result = top_bigrams(user_input)
            
            if bigrams_result:  # Check if bigrams list is not empty
                bigrams_df = pd.DataFrame(bigrams_result, columns=['Bigram', 'Count'])
                bigrams_df['Bigram'] = bigrams_df['Bigram'].apply(lambda x: ", ".join(x))
                st.table(bigrams_df)
            else:
                st.write("No bigrams found!")

            # Top 10 Trigrams
            trigrams_result = top_trigrams(user_input)
            st.subheader("Top 10 Trigrams (Group of 3 words)")
            if trigrams_result:
                trigrams_df = pd.DataFrame(trigrams_result, columns=['Trigram', 'Count'])
                trigrams_df['Trigram'] = trigrams_df['Trigram'].apply(lambda x: ", ".join(x))
                st.table(trigrams_df)
            else:
                st.write("No Trigrams found")

            # Display Top 5 Topics (Using LDA)
            # st.subheader("Top 5 Topics (Using LDA)")
            # topics = lda_topics(user_input)
            
            # if topics:
            #     for idx, topic in enumerate(topics, 1):
            #         st.write(f"Topic {idx}: {topic}")
            # else:
            #     st.write("No topics found!")

            # Create a placeholder for the loading message
            loading_message = st.empty()
            loading_message.text("Loading...")

             # Perform the topic analysis
            topics = bert_topics(user_input)

            # Overwrite the loading message
            loading_message.text("")  # This will effectively hide the "Loading..." message

            st.subheader("Topics using BERT")
            if topics:
                for i, topic in enumerate(topics, 1):
                    st.write(f"Topic {i}: {topic}")
            else: 
                st.write("No topics found")   





