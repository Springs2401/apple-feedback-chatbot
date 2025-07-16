import pandas as pd
import numpy as np
import re, string
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning
def clean_tweets(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[%s]' % re.escape(string.punctuation), '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'\s{2,}', ' ', tweet)
    return tweet

def preprocess_tweet(tweet):
    words = tweet.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load data and train model
df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='ISO-8859-1')
df = df[['sentiment', 'text']].dropna()
df = df[df['sentiment'] != 'not_relevant']
df['sentiment'] = df['sentiment'].astype(int)
df['sentiment'] = df['sentiment'].map({1: 0, 3: 1, 5: 2})
df['text'] = df['text'].apply(lambda x: x.replace('\n1', '').replace('\n', ' '))
df['clean_text'] = df['text'].apply(clean_tweets)
df['processed_text'] = df['clean_text'].apply(preprocess_tweet)

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['processed_text'])
y = df['sentiment']
model = LogisticRegression().fit(X, y)

# Sentiment prediction
def prepare_message(message):
    message = clean_tweets(message)
    message = preprocess_tweet(message)
    return message

def predict_sentiment(message):
    vector = tfidf.transform([prepare_message(message)])
    return model.predict(vector)[0]

# Load LLM
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Generate reply
def generate_reply(user_message, sentiment_label):
    sentiment_map = {
        0: "The user is unhappy. Apologize politely and offer help.",
        1: "The user is neutral. Thank them and invite further feedback.",
        2: "The user is happy. Thank them warmly and show appreciation."
    }
    system_prompt = sentiment_map.get(sentiment_label, "Respond politely to the user's feedback.")
    full_prompt = f"<|user|>\n{system_prompt}\n\nUser: {user_message}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(llm_model.device)
    attention_mask = inputs["attention_mask"].to(llm_model.device)
    output = llm_model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

def chat_bot(message, history):
    try:
        cleaned = prepare_message(message)
        sentiment = predict_sentiment(cleaned)
        reply = generate_reply(message, sentiment)
        label = {0: "Negative", 1: "Neutral", 2: "Positive"}[sentiment]
        full_reply = f"(Sentiment: {label}) {reply}"
        return full_reply
    except Exception as e:
        return f"Error: {str(e)}"


iface = chat = gr.ChatInterface(
    fn=chat_bot,
    title="Welcome to the Apple Feedback Assistant 👋🏼",
    description="Type your feedback below (or type 'Exit' to quit)",
    theme="soft",
)

chat.launch()


iface.launch()
