
import streamlit as st
import requests
from typing import Generator
from groq import Groq
import random

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Set page config
st.set_page_config(page_icon="💬", layout="wide",
                   page_title="VEHI-Defect: Vehicle Defect Transparency Tool")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 80px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🏎️")

st.subheader("VEHI-Defect: Vehicle Defect Transparency Tool", divider="rainbow", anchor=False)


# Function to get data from the NHTSA API
def get_data(make, model, modelYear):
    url = f"https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={modelYear}"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        myjson = response.json()

        summaries = []
        for result in myjson['results']:
            summaries.append({'summary': result['summary']})

        num_summaries = min(len(summaries), 5)  # Limit to 5 summaries
        sampled_summaries = random.sample(summaries, num_summaries)

        return sampled_summaries

    else:
        st.error("Failed to get data from the API.")

# Load fine-tuned T5 model and tokenizer
model_name = "ItsMayur/t5-small-finetuned-vehidefecttwo"  # e.g., "t5-small-finetuned-vehidefect"
t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function for text summarization
def generate_summary(complaint):
    input_text = complaint
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    with torch.no_grad():
        summary_ids = t5_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True, pad_token_id=tokenizer.pad_token_id)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

################################################################################################################################

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

model_option = "mixtral-8x7b-32768"

max_tokens_range = models[model_option]["tokens"]


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content




# Get user inputs
make = st.text_input("Enter the make: ", "")
model = st.text_input("Enter the model: ", "")
modelYear = st.text_input("Enter the model year: ", "")

# Send API call
if st.button("Get Data"):
    if make != "" and model != "" and modelYear != "":
        data = get_data(make, model, modelYear)

        if data:
            for complaint in data:
                # Fetch response from Groq API
                try:
                    chat_completion = client.chat.completions.create(
                        model=model_option,
                        messages=[
                            {
    "role": "system",
    "content": f"Analyze the following customer complaint:\n\n{complaint['summary']}\n\n**Extract the following information:**\n\n1. **Vehicle component(s) mentioned:** Identify and list all the specific vehicle components mentioned in the complaint. (e.g., brakes, engine, transmission)\n2. **Relationship between customer and manufacturer/dealership:** Based on the complaint, determine the current relationship between the customer and the manufacturer/dealership. (e.g., seeking repair, dissatisfied with service, positive experience)\n3. **Severity Score:** Based on the details of the complaint, assign a severity score from 1 (least severe) to 5 (most severe). Consider factors such as the potential risk to safety, cost of repairs, and impact on the vehicle's functionality."
}
                        ],
                        max_tokens=max_tokens_range,
                        stream=True
                    )

                    # Use the generator function with st.write_stream
                    with st.chat_message("user", avatar="👨‍💻"):
                        st.markdown(f"**Customer Complaint:**\n\n{complaint['summary']}\n")
                    
                    with st.spinner("Analyzing..."):
                        chat_responses_generator = generate_chat_responses(chat_completion)
                        full_response = list(chat_responses_generator)
                        
                    # Display the assistant's response with proper formatting
                    if full_response:
                        st.write("".join(full_response))
                    else:
                        st.warning("No analysis found.")
                        
                    # Generate summary for the complaint
                    summary = generate_summary(complaint['summary'])  # Pass complaint text, not the dictionary

                    # Display the summary
                    with st.expander("Summary"):
                        st.write(summary)
                        
                except Exception as e:
                    st.error(e, icon="🚨")

        else:
            st.warning("No data found.")
    else:
        st.warning("All fields must be filled.")

