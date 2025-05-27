import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
from transformers import pipeline
import networkx as nx

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Initialize the transformers pipeline for abstractive summarization
        self.abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def read_article(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def sentence_similarity(self, sent1, sent2):
        words1 = [word.lower() for word in word_tokenize(sent1) if word.lower() not in self.stop_words]
        words2 = [word.lower() for word in word_tokenize(sent2) if word.lower() not in self.stop_words]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for word in words1:
            vector1[all_words.index(word)] += 1
        
        for word in words2:
            vector2[all_words.index(word)] += 1
            
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 != idx2:
                    similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2])
                    
        return similarity_matrix

    def extractive_summarize(self, text, num_sentences=3):
        """
        Generate extractive summary by ranking sentences using PageRank algorithm
        """
        sentences = self.read_article(text)
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
            
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        # Calculate sentence scores using PageRank algorithm
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sort sentences by score and select top n
        ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
        summary_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
        
        # Sort sentences by their original order
        summary_sentences.sort(key=lambda x: sentences.index(x))
        
        return " ".join(summary_sentences)

    def abstractive_summarize(self, text, max_length=130, min_length=30):
        """
        Generate abstractive summary using BART model
        """
        summary = self.abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def summarize(self, text, method='both', num_sentences=3):
        """
        Generate summary using specified method (extractive, abstractive, or both)
        """
        if method.lower() == 'extractive':
            return {'extractive': self.extractive_summarize(text, num_sentences)}
        elif method.lower() == 'abstractive':
            return {'abstractive': self.abstractive_summarize(text)}
        else:
            return {
                'extractive': self.extractive_summarize(text, num_sentences),
                'abstractive': self.abstractive_summarize(text)
            } 