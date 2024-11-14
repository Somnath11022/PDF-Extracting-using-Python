# Required Libraries
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from textblob import TextBlob

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab') 

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    pdf_text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    return pdf_text

# Function to preprocess and tokenize text
def preprocess_text(text):
    # Lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize sentences
    sentences = sent_tokenize(text)
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word not in stop_words]
                # Shorten sentence to relevant words
        processed_sentences.append(' '.join(filtered_words[:200]))
    return processed_sentences

# Function to categorize sentences based on keywords
def categorize_text(sentences):
    categories = {
        'Future Planning': ['plan', 'strategy', 'future', 'goal', 'vision'],
        'Achievements': ['achievement', 'award', 'success', 'milestone'],
        'Positive Aspects': ['good', 'excellent', 'positive', 'benefit', 'advantage']
    }
    categorized_data = defaultdict(list)

    for sentence in sentences:
        for category, keywords in categories.items():
            if any(keyword in sentence for keyword in keywords):
                categorized_data[category].append(sentence)
                break  # To avoid double categorization
    return categorized_data

# Function to perform sentiment analysis
def analyze_sentiment(sentences):
    sentiment_results = []
    for sentence in sentences:
        analysis = TextBlob(sentence)
        if analysis.sentiment.polarity > 0.1:
            sentiment_results.append((sentence, 'Positive'))
    return sentiment_results

# Main workflow
file_path = 'SJS Transcript Call.pdf'  # Replace with your PDF file path
text = extract_text_from_pdf(file_path)
processed_sentences = preprocess_text(text)
categorized_data = categorize_text(processed_sentences)
sentiment_data = analyze_sentiment(processed_sentences)

# Display categorized data
for category, sentences in categorized_data.items():
    print(f"\nCategory: {category}")
    for sentence in sentences:
        print(f"- {sentence}")

# Display sentences with positive sentiment
print("\nPositive Sentiment Sentences:")
for sentence, sentiment in sentiment_data:
    print(f"- {sentence} ({sentiment})")


# Save the output to a text file
with open('output.txt', 'w', encoding='utf-8') as output_file:
    for category, sentences in categorized_data.items():
        output_file.write(f"\nCategory: {category}\n")
        for sentence in sentences:
            output_file.write(f"{sentence}\n")

    output_file.write("\nSentiment Analysis Results:\n")
    for sentence, sentiment in sentiment_data:
        output_file.write(f"{sentence} - {sentiment}\n")

print("Output saved to output.txt")