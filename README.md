# AI-Powered Feedback and Complaints Chatbot for Apple Brand Monitoring

***Try It Live:** https://huggingface.co/spaces/Springs24/apple-feedback-chatbot*

## Overview

This project introduces a smart AI chatbot that receives customer feedback about Apple products, understands how the user is feeling (positive, negative, or neutral), and provides a polite, helpful response instantly. It combines sentiment analysis and generative AI to act like a mini customer support agent. It detects the mood of the message and replies appropriately to complaints, suggestions, or appreciation.

### Key Features:

- Accepts live customer feedback or complaints as input  
- Detects sentiment using a trained ML model (TF-IDF + Logistic Regression)  
- Generates polite, brand-appropriate responses using a language model  
- Supports Apple’s customer service by automating early response  
- Built and deployed using Gradio and Hugging Face Spaces

## Why This Matters

Today, customers expect brands to respond quickly—especially on platforms like social media. Apple receives large volumes of public feedback daily, which can be hard to manage manually.

This chatbot helps reduce that pressure by automatically understanding the user's mood and sending a quick, polite reply using generative AI. It can be expanded to websites, apps, or customer service dashboards to improve experience and efficiency.

## How It Works

1. A user types a message into the chatbot.
2. The message is cleaned and preprocessed using Python (NLTK, regex).
3. A trained sentiment analysis model classifies it as Positive, Neutral, or Negative.
4. Based on the sentiment, a polite response is generated using a large language model.
5. The chatbot shows this reply instantly.
6. Optionally, the interaction can be logged for future analysis.

### Workflow:

User Input → Preprocessing → TF-IDF → Sentiment Model → Sentiment Label → Generative Module → Chatbot Reply

## Tech Stack

- Python
- Pandas, NumPy, Scikit-learn, NLTK
- TF-IDF Vectorizer, Logistic Regression
- Hugging Face Transformers (DeepSeek LLM)
- Gradio for deployment
- Hugging Face Spaces as hosting platform
