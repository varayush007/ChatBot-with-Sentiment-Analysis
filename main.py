import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import re
import os
from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    memory=ConversationBufferWindowMemory(k=2),
)

sentiment_pipeline = pipeline("sentiment-analysis",
                              model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


def main():
    st.title("Conversational Chatbot")

    if "bot_responses" not in st.session_state:
        st.session_state.bot_responses = []

    user_input = st.text_input("Type question to start chatting or type quit to exit")

    if st.button("Send"):
        if user_input.lower() == 'quit':
            st.write("Chatbot: Goodbye!")
        else:
            raw_response = conversation_with_summary.predict(input=user_input)
            response = re.split(r'\n|AI:', raw_response)[0].strip()
            if "Human:" in response:
                response = response.split("Human:")[0].strip()

            st.session_state.bot_responses.append(response)
            st.write("Chatbot:", response)

    def perform_sentiment_analysis(answers):
        elements = []
        for it in answers:
            sentiment_result = sentiment_pipeline(it)
            sentiment_label = sentiment_result[0]['label']
            elements.append(sentiment_label)
        return elements

    def visualize_sentiment_analysis(elements):
        sentiment_counts = {label: elements.count(label) for label in set(elements)}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=140,
                colors=['green', 'red', 'yellow'])
        ax1.set_title('Sentiment Analysis (Pie Chart)')

        ax2.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'yellow'])
        ax2.set_title('Sentiment Analysis (Bar Graph)')
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Count')

        st.pyplot(fig)

    if st.sidebar.button("Perform Sentiment Analysis"):
        if st.session_state.bot_responses:
            sentiments = perform_sentiment_analysis(st.session_state.bot_responses)
            st.write("Sentiments:", sentiments)
        else:
            st.write("No bot responses available for sentiment analysis.")

    if st.sidebar.button("Visualize Sentiment Analysis"):
        if st.session_state.bot_responses:
            sentiments = perform_sentiment_analysis(st.session_state.bot_responses)
            visualize_sentiment_analysis(sentiments)
        else:
            st.write("No bot responses available for visualization.")


if __name__ == "__main__":
    main()
