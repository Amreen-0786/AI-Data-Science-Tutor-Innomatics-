import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your Google Generative AI API key
api_key = "AIzaSyD7ocrTyoMnaBsi7RpTSvV4G9NIeUDlmdk"  # Replace with your actual API key

# Initialize the Gemini models
def get_model(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        google_api_key=api_key
    )

# Add memory for conversation awareness
memory = ConversationBufferMemory()

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "topic"],
    template="""
    You are a Data Science Tutor. Your task is to help users resolve their data science doubts.
    The user has selected the topic: {topic}. Focus your answers on this topic.
    Provide clear, concise, and accurate explanations. If the question is not related to data science,
    politely inform the user that you can only assist with data science topics.

    User: {input}
    Tutor:
    """
)

# Function to generate visualizations
def generate_visualization(df, query):
    try:
        if "bar graph" in query.lower():
            column = query.lower().split("bar graph")[1].strip()
            if column in df.columns:
                st.write(f"### Bar Graph for {column}")
                plt.figure(figsize=(10, 6))
                sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values)
                plt.xticks(rotation=45)
                st.pyplot(plt)
            else:
                st.error(f"Column '{column}' not found in the dataset.")
        elif "histogram" in query.lower():
            column = query.lower().split("histogram")[1].strip()
            if column in df.columns:
                st.write(f"### Histogram for {column}")
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], kde=True)
                st.pyplot(plt)
            else:
                st.error(f"Column '{column}' not found in the dataset.")
        else:
            st.warning("Unsupported visualization type. Please ask for a bar graph or histogram.")
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

# Streamlit app
def main():
    st.title("🌟 AI Data Science Tutor 🌟")
    st.write("Welcome to the AI Data Science Tutor! Ask me anything about data science.")

    # Sidebar for model and topic selection
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Choose a model",
            ["gemini-1.0-pro", "gemini-1.5-pro"],
            index=1
        )
        topic = st.selectbox(
            "Choose a topic",
            ["Machine Learning", "Deep Learning", "Statistics", "Data Visualization", "General Data Science"],
            index=0
        )

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # File upload for data analysis
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("### Dataset Preview")
        st.sidebar.write(df.head())

    # User input
    user_input = st.text_input("Ask your data science question:")

    if user_input:
        # Initialize the selected model
        llm = get_model(model_name)
        conversation = ConversationChain(llm=llm, memory=memory)

        # Generate response using the conversation chain
        response = conversation.run(prompt_template.format(input=user_input, topic=topic))

        # Update conversation history
        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Tutor: {response}")

        # Handle data visualization requests
        if topic == "Data Visualization" and df is not None:
            generate_visualization(df, user_input)

    # Display conversation history
    st.write("### Conversation History")
    for message in st.session_state.conversation_history:
        st.write(message)

    # Export conversation history
    if st.button("Export Conversation History"):
        conversation_text = "\n".join(st.session_state.conversation_history)
        st.download_button(
            label="Download Conversation",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain"
        )

# Run the app
if __name__ == "__main__":
    main()