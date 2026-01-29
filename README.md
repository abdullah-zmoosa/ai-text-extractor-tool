# AI Text Extractor (Vision IQ)

An intelligent web application that uses **Gemma 3 27b** (multimodal vision-language model) to extract text from images, categorize them automatically, and allow for smart visual querying.

## Features

- **Multimodal Extraction**: Uses Gemma 3's visual reasoning to read text and understand image context.
- **Smart Categorization**: Automatically detects if your upload is a Receipt, Business Card, Handwritten Note, or Invoice.
- **Live Previews**: See your images and extracted text side-by-side as they process.
- **Editable Results**: Fix any minor AI typos directly in the app before exporting.
- **Bulk Processing**: Upload multiple images and let the app handle the rate limits automatically.
- **Multi-Format Export**: Download your data as **CSV**, **JSON**, or a **ZIP** archive of individual text files.
- **Quick Templates**: One-click instructions for common tasks like "Translate to English" or "Format as JSON".

## Prerequisites

- Python 3.11
- A Google Gemini/Gemma API Key from [Google AI Studio](https://aistudio.google.com/).

## Installation 📦

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdullah-zmoosa/ai-text-extractor-tool.git
   cd ai-text-extractor-tool
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Getting your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Log in with your Google Account.
3. Click on the **"Get API key"** button on the left sidebar.
4. Click **"Create API key"**. You can choose to create it in a new or existing Google Cloud project.
5. Copy the generated key.
6. In the app, you can check **"Save API Key locally"** to store it in a `.env` file for future sessions.

## Project Structure

- `app.py`: The main Streamlit application logic.
- `requirements.txt`: Python package dependencies.
- `.env`: (Auto-generated) Stores your API key securely if saved.

## Usage Tips

- **Rate Limits**: The app includes a continuous throttler (~15 requests per minute) to ensure you stay within free tier limits.
- **Custom Prompts**: Use the "Custom Instructions" box to tell the AI exactly what you want (e.g., "Translate to Arabic" or "Extract only the dates etc.").
