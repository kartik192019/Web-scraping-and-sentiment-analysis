import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Function to scrape a website and extract data
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', class_='entry-title').text.strip()
        content = soup.find('div', class_='td-post-content tagdiv-type').text.strip()
        return title, content
    else:
        return None, None

# Function to clean text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return cleaned_tokens

# Function to create dictionaries of positive and negative words
def create_dictionary(file_path, sentiment_score):
    word_dict = {}
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            word = line.strip()
            word_dict[word] = sentiment_score
    return word_dict

# Function to calculate scores
def calculate_scores(text, pos_dict, neg_dict):
    positive_score = sum(pos_dict.get(word, 0) for word in text)
    negative_score = sum(neg_dict.get(word, 0) for word in text)
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    polarity_score = max(min(abs(polarity_score), 1), -1)
    subjectivity_score = (positive_score + negative_score) / (len(text) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to count syllables in a word
def count_syllables(word):
    vowels = 'aeiouy'
    count = 0
    previous_is_vowel = False
    for char in word.lower():
        if char in vowels and not previous_is_vowel:
            count += 1
            previous_is_vowel = True
        elif char not in vowels:
            previous_is_vowel = False
    if word.endswith('e') and count > 1:
        count -= 1
    count = max(count, 1)
    return count

# Function to calculate readability using Gunning Fog index
def calculate_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    num_complex_words = sum(1 for word in filtered_words if count_syllables(word) > 2)
    fog_index = 0.4 * (avg_sentence_length + (num_complex_words / len(words)) * 100)
    return avg_sentence_length, num_complex_words, fog_index

# Function to count cleaned words in the text
def count_cleaned_words(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    return len(cleaned_words)

# Function to calculate count of personal pronouns
def count_personal_pronouns(text):
    pattern = r'\b(?:I|we|my|ours|us)\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return len(matches)

# Function to calculate average word length
def calculate_average_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    average_word_length = total_characters / total_words if total_words > 0 else 0
    return average_word_length

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def calculate_syllable_count_per_word(text):
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    
    # Calculate syllable count per word
    syllable_count_per_word = [count_syllables(word) for word in cleaned_words]
    
    return syllable_count_per_word

# List of website URLs to scrape
urls = [
    "https://insights.blackcoffer.com/rising-it-cities-and-its-impact-on-the-economy-environment-infrastructure-and-city-life-by-the-year-2040-2/",
    # Add more URLs here
]

# Initialize lists to store scraped data
titles = []
contents = []
positive_scores = []
negative_scores = []
polarity_scores = []
subjectivity_scores = []
avg_sentence_lengths = []
num_complex_words_list = []
fog_indices = []
word_counts = []
syllable_counts_per_word_list = []
personal_pronouns_counts = []
average_word_lengths = []
sentiment_scores_list = []

# Load Positive and Negative dictionaries
positive_dict = create_dictionary('MasterDictionary/positive-words.txt', 1)
negative_dict = create_dictionary('MasterDictionary/negative-words.txt', -1)

# Iterate through each URL and scrape data
for url in urls:
    title, content = scrape_website(url)
    if title and content:
        # Clean the text
        cleaned_text = clean_text(content)
        
        # Calculate scores
        positive_score, negative_score, polarity_score, subjectivity_score = calculate_scores(cleaned_text, positive_dict, negative_dict)
        avg_sentence_length, num_complex_words, fog_index = calculate_readability(content)
        total_cleaned_words = count_cleaned_words(content)
        syllable_counts_per_word = calculate_syllable_count_per_word(content)
        personal_pronouns_count = count_personal_pronouns(content)
        average_word_length = calculate_average_word_length(content)
        sentiment_scores = analyze_sentiment(content)
        
        # Append scraped data to lists
        titles.append(title)
        contents.append(content)
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
        polarity_scores.append(polarity_score)
        subjectivity_scores.append(subjectivity_score)
        avg_sentence_lengths.append(avg_sentence_length)
        num_complex_words_list.append(num_complex_words)
        fog_indices.append(fog_index)
        word_counts.append(total_cleaned_words)
        syllable_counts_per_word_list.append(syllable_counts_per_word)
        personal_pronouns_counts.append(personal_pronouns_count)
        average_word_lengths.append(average_word_length)
        sentiment_scores_list.append(sentiment_scores)

# Create a DataFrame to store the scraped data
df = pd.DataFrame({
    'Title': titles,
    'Content': contents,
    'Positive Score': positive_scores,
    'Negative Score': negative_scores,
    'Polarity Score': polarity_scores,
    'Subjectivity Score': subjectivity_scores,
    'Average Sentence Length': avg_sentence_lengths,
    'Number of Complex Words': num_complex_words_list,
    'Fog Index': fog_indices,
    'Word Count': word_counts,
    'Syllable Count Per Word': syllable_counts_per_word_list,
    'Personal Pronouns Count': personal_pronouns_counts,
    'Average Word Length': average_word_lengths,
    'Sentiment Scores': sentiment_scores_list
})

# Save the DataFrame to a CSV file
df.to_csv('scraped_data.csv', index=False)

