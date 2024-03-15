import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from nltk.sentiment import SentimentIntensityAnalyzer


url = "https://insights.blackcoffer.com/rising-it-cities-and-its-impact-on-the-economy-environment-infrastructure-and-city-life-by-the-year-2040-2/"


response = requests.get(url)


if response.status_code == 200:

    soup = BeautifulSoup(response.text, 'html.parser')
    
  
    title = soup.find('h1', class_='entry-title').text.strip()
    content = soup.find('div', class_='td-post-content tagdiv-type').text.strip()
    

    print("Title:", title)
    
else:
    print("Failed to retrieve the webpage")



# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1.1: Cleaning using Stop Words Lists
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return cleaned_tokens

# Step 1.2: Creating a dictionary of Positive and Negative words
def create_dictionary(file_path, sentiment_score):
    word_dict = {}
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            word = line.strip()
            word_dict[word] = sentiment_score
    return word_dict




# Step 1.3: Extracting Derived variables
def calculate_scores(text, pos_dict, neg_dict):
    positive_score = sum(pos_dict.get(word, 0) for word in text)
    negative_score = sum(neg_dict.get(word, 0) for word in text)
    # Calculate Polarity Score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)

# Ensure Polarity Score is within range [-1, 1]
    polarity_score = max(min(abs(polarity_score), 1), -1)

    subjectivity_score = (positive_score + negative_score) / (len(text) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Load Positive and Negative dictionaries
positive_dict = create_dictionary('MasterDictionary/positive-words.txt', 1)
negative_dict = create_dictionary('MasterDictionary/negative-words.txt', -1)



# Clean the text
cleaned_text = clean_text(content)

# Calculate scores
positive_score, negative_score, polarity_score, subjectivity_score = calculate_scores(cleaned_text, positive_dict, negative_dict)

# Print results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)






nltk.download('stopwords')

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
    # Adjust for words ending with 'e' (not counted as a separate syllable)
    if word.endswith('e') and count > 1:
        count -= 1
    # Ensure at least one syllable for short words
    count = max(count, 1)
    return count

# Function to calculate readability using Gunning Fog index
def calculate_readability(text):
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Calculate Average Sentence Length
    avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
    
    # Remove stopwords for complex words calculation
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    
    # Calculate number of complex words
    num_complex_words = sum(1 for word in filtered_words if count_syllables(word) > 2)
    
    # Calculate Fog Index
    fog_index = 0.4 * (avg_sentence_length + (num_complex_words / len(words)) * 100)
    
    return avg_sentence_length, num_complex_words, fog_index


# Calculate readability
avg_sentence_length, num_complex_words, fog_index = calculate_readability(content)
# Print results
print("Gunning Fog Index:", fog_index)
print("Average Sentence Length:", avg_sentence_length)
print("Number of Complex Words:", num_complex_words)




# Function to count cleaned words in the text
def count_cleaned_words(text):
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    
    return len(cleaned_words)


# Calculate total cleaned words
total_cleaned_words = count_cleaned_words(content)

# Print result
print("word count:", total_cleaned_words)



def calculate_syllable_count_per_word(text):
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    
    # Calculate syllable count per word
    syllable_count_per_word = [count_syllables(word) for word in cleaned_words]
    
    return syllable_count_per_word



# Calculate syllable count per word
syllable_count_per_word = calculate_syllable_count_per_word(content)

# Print result
print("Syllable Count Per Word:", syllable_count_per_word)




# Function to calculate count of personal pronouns
def count_personal_pronouns(text):
    # Define personal pronouns pattern using regex
    pattern = r'\b(?:I|we|my|ours|us)\b'
    
    # Find all matches of personal pronouns in the text
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
    # Count the number of matches
    count = len(matches)
    
    return count



# Calculate count of personal pronouns
personal_pronouns_count = count_personal_pronouns(content)

# Print result
print("Count of Personal Pronouns:", personal_pronouns_count)




# Function to calculate average word length
def calculate_average_word_length(text):
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Calculate total number of characters in all words
    total_characters = sum(len(word) for word in words)
    
    # Calculate total number of words
    total_words = len(words)
    
    # Calculate average word length
    if total_words > 0:
        average_word_length = total_characters / total_words
    else:
        average_word_length = 0
    
    return average_word_length



# Calculate average word length
average_word_length = calculate_average_word_length(content)

# Print result
print("Average Word Length:", average_word_length)




# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    # Analyze sentiment of the text
    sentiment_scores = sia.polarity_scores(text)
    
    return sentiment_scores



# Perform sentiment analysis
sentiment_scores = analyze_sentiment(content)

# Print result
print("Sentiment Scores:", sentiment_scores)
