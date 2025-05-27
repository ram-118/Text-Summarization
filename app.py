from flask import Flask, render_template, request, jsonify
from summarizer import TextSummarizer

app = Flask(__name__)
summarizer = TextSummarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '')
    method = data.get('method', 'both')
    num_sentences = int(data.get('num_sentences', 3))
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        summary = summarizer.summarize(text, method=method, num_sentences=num_sentences)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 