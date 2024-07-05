

This [Streamlit](https://streamlit.io/) app integrates with the [Groq API](https://groq.com/) to provide a chat interface where users can interact with advanced language models. It allows users to choose between two models for generating responses, enhancing the flexibility and user experience of the chat application.

It is blazing FAST, try it and see! 🏎️ 💨 💨 💨


## Requirements

- Streamlit
- Groq Python SDK
- Python 3.7+

## Setup and Installation

- **Install Dependencies**:

  ```bash
  pip install streamlit groq
  ```

- **Set Up Groq API Key**:

  Ensure you have an API key from Groq. This key should be stored securely using Streamlit's secrets management:

  ```toml
  # .streamlit/secrets.toml
  GROQ_API_KEY="your_api_key_here"
  ```

- **Run the App**:
  Navigate to the app's directory and run:

```bash
streamlit run streamlit_app.py
```

## Usage


## Contributing

Contributions to enhance the app, fix bugs, or improve documentation are welcome.

Please feel free to fork the repository, make your changes, and submit a pull request.
