CodeAlpha - AI FAQ Chatbot

An intelligent desktop chatbot application built with Python, CustomTkinter, and Scikit-Learn. It processes natural language queries using TF-IDF Vectorization and Cosine Similarity to match user questions with stored FAQ responses in real time.

Developed as part of the CodeAlpha Artificial Intelligence Internship.

🌟 Key Features

Natural Language Processing (NLP): Uses scikit-learn's TF-IDF Vectorization and Cosine Similarity to understand and match intent, even with varied phrasing.

Smart Fallback & Greeting Handling: Gracefully handles basic greetings ("hi", "hello") and provides helpful fallback guidance for unmatched queries.

Modern GUI: Clean, dark-mode desktop chat interface powered by CustomTkinter.

Interactive UI: Smooth chat history display, real-time message scrolling, and Enter-key binding for quick sending.

🛠️ Tech Stack & Requirements

Python 3.8+

GUI Framework: customtkinter

NLP & Machine Learning: scikit-learn, nltk

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/muqtasidtalha/CodeAlpha_ChatbotForFAQs.git
cd CodeAlpha_ChatbotForFAQs


2. Install Dependencies

pip install -r requirements.txt


3. Run the Application

python app.py


📁 Project Structure

CodeAlpha_ChatbotForFAQs/
├── app.py              # Main GUI and NLP FAQ matching engine
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
