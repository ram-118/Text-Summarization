# Text-Summarization



# Text Summarization Application

This is a powerful text summarization application that provides both extractive and abstractive summarization capabilities using state-of-the-art NLP techniques.

## Features

- **Extractive Summarization**: Uses TextRank algorithm to extract the most important sentences from the text
- **Abstractive Summarization**: Utilizes BART model to generate concise summaries
- **Web Interface**: Modern and responsive UI built with Flask and Tailwind CSS
- **Flexible Options**: Choose between extractive, abstractive, or both summarization methods

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd text-summarization
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter your text in the input field

4. Choose your preferred summarization method:
   - **Both**: Get both extractive and abstractive summaries
   - **Extractive**: Get only the extractive summary
   - **Abstractive**: Get only the abstractive summary

5. For extractive summarization, you can specify the number of sentences to include in the summary

6. Click the "Summarize" button to generate the summary

## Technical Details

### Extractive Summarization
- Uses NLTK for text processing
- Implements TextRank algorithm using NetworkX
- Removes stopwords and performs sentence tokenization
- Calculates sentence similarity using cosine distance

### Abstractive Summarization
- Uses the BART model from Facebook/Meta
- Implements using the Hugging Face Transformers library
- Provides concise, human-readable summaries

## Requirements

- Python 3.7+
- NLTK
- PyTorch
- Transformers
- Flask
- NetworkX
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
