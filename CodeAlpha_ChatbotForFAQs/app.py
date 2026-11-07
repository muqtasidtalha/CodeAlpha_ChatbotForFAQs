import customtkinter as ctk
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resource
nltk.download('punkt', quiet=True)

# Pre-defined FAQ Dataset (Questions and Answers)
FAQ_DATA = [
    {
        "question": "hello hi hey greetings good morning good afternoon",
        "answer": "Hello! How can I assist you today with the CodeAlpha Internship?"
    },
    {
        "question": "What is CodeAlpha?",
        "answer": "CodeAlpha is a software development company that provides internships and hands-on experience in emerging technologies like AI, Web Development, and Cybersecurity."
    },
    {
        "question": "How do I submit my completed internship tasks?",
        "answer": "You must push your code to a public GitHub repository, post a video demo on LinkedIn tagging @CodeAlpha, and submit the link through the official Google Form."
    },
    {
        "question": "How many tasks do I need to complete to get a certificate?",
        "answer": "You need to complete a minimum of 2 or 3 tasks from your assigned domain to be eligible for the internship certificate."
    },
    {
        "question": "What perks do interns get upon completion?",
        "answer": "Interns receive a QR-verified Completion Certificate, an Offer Letter, a Unique ID Certificate, a Letter of Recommendation (performance-based), and resume support."
    },
    {
        "question": "What should be the naming convention for GitHub repositories?",
        "answer": "Your repository should be named strictly in the format: CodeAlpha_ProjectName (e.g., CodeAlpha_ChatbotForFAQs)."
    },
    {
        "question": "Where can I contact CodeAlpha support?",
        "answer": "You can reach out via email at services@codealpha.tech or contact support on WhatsApp at +91 9336576683."
    }
]

# Extract questions for NLP matching
questions = [faq["question"] for faq in FAQ_DATA]

# Initialize TF-IDF Vectorizer with char/word n-grams for better short-word matching
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(questions)


def get_best_response(user_query, threshold=0.15):
    """Calculates cosine similarity to find the closest FAQ match."""
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return FAQ_DATA[best_idx]["answer"]
    else:
        return "I'm sorry, I couldn't find an answer to that specific question. Try asking about internship perks, task submission rules, or support contacts!"


# GUI Application Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FAQChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CodeAlpha - FAQ Chatbot")
        self.geometry("600x650")
        self.resizable(False, False)

        self._build_ui()

    def _build_ui(self):
        # Header
        title_label = ctk.CTkLabel(
            self, 
            text="CodeAlpha FAQ Assistant", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(pady=(15, 10))

        # Chat History Window
        self.chat_display = ctk.CTkTextbox(
            self, 
            width=540, 
            height=460, 
            font=("Consolas", 12),
            state="disabled"
        )
        self.chat_display.pack(padx=20, pady=10)

        # Welcome Message
        self.display_message("Bot", "Hello! Ask me anything about the CodeAlpha Internship program.")

        # Input Frame
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=20, pady=10)

        self.user_input = ctk.CTkEntry(
            input_frame, 
            placeholder_text="Type your question here...", 
            width=430,
            height=38
        )
        self.user_input.pack(side="left", padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_btn = ctk.CTkButton(
            input_frame, 
            text="Send", 
            width=90, 
            height=38,
            command=self.send_message
        )
        self.send_btn.pack(side="right")

    def display_message(self, sender, text):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {text}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def send_message(self):
        query = self.user_input.get().strip()
        if not query:
            return

        # Display user question
        self.display_message("You", query)
        self.user_input.delete(0, "end")

        # Get bot response using NLP cosine similarity
        bot_response = get_best_response(query)
        self.display_message("Bot", bot_response)


if __name__ == "__main__":
    app = FAQChatbotApp()
    app.mainloop()