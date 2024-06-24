from flask import Flask, jsonify, request
import pandas as pd
import textdistance
import re
from collections import Counter
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import emoji
import gzip
app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model("Lnew.keras", compile=False)

# Define the emoji dictionary
emoji_dict = {
    '0': ':beating_heart:',
    '1': ':baseball:',
    '2': ':face_with_tears_of_joy:',
    '3': ':confounded_face:',
    '4': '🍽️'
}

# Load the embedding matrix (make sure this is the correct path to your embeddings)
embedding_matrix = {}  # Load your actual embedding matrix here
with gzip.open('glove.6B.50d.txt.gz', 'rt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        emb = np.array(values[1:], dtype='float32')
        embedding_matrix[word] = emb
    
    embedding_matrix[word] = emb
def preprocess_input_text(input_text, embedding_matrix, max_len=10):
    embedding_data = np.zeros((1, max_len, 50))
    words = input_text.split()
    
    for i, word in enumerate(words):
        if i < max_len:
            embedding_vector = embedding_matrix.get(word.lower())
            if embedding_vector is not None:
                embedding_data[0, i] = embedding_vector
                
    return embedding_data

#-------------------------------------------------------------------------------------------------------------------------------------------------------


words = []

with open('autocorrect_book.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)  # converting into array of words 
    words += words

V = set(words)  # only unique words remains
words_freq_dict = Counter(words)  # {the: 4564, is: 5779} means 'the' came 4564 times in the book we gave
Total = sum(words_freq_dict.values())  # total number of words in the book we gave
probs = {}  # empty dictionary

for k in words_freq_dict.keys():
    probs[k] = words_freq_dict[k] / Total  # calculating probability of every word (freq_of_that_word / Total)

@app.route('/suggest', methods=['POST'])
def suggest():
    keyword = request.json['keyword'].strip()  # expecting JSON data
    if keyword:
        keyword = keyword.lower()
        
        # Handle single-letter keyword separately
        if len(keyword) == 1:
            # Filter words that start with the single letter
            filtered_words = {word: prob for word, prob in probs.items() if word.startswith(keyword)}
            similarities = [1] * len(filtered_words)  # Since all words start with the single letter, they are maximally similar
            df = pd.DataFrame.from_dict(filtered_words, orient='index').reset_index()
            df.columns = ['Word', 'Prob']
            df['Similarity'] = similarities
        else:
            similarities = [1 - textdistance.Jaccard(qval=2).distance(v, keyword) for v in words_freq_dict.keys()]  # Jaccard calculates dissimilarities so we subtract one to get similarity
            df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
            df.columns = ['Word', 'Prob']
            df['Similarity'] = similarities
        
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False)[['Word', 'Similarity']]  # Sort the DataFrame by the 'Similarity' and 'Prob' columns in descending order.
        suggestions_list = suggestions.head(5).to_dict('records')  # Get top 5 suggestions
        return jsonify(suggestions=suggestions_list)
    return jsonify(suggestions=[])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
   
    input_text = data['text']
    
    X_input = preprocess_input_text(input_text, embedding_matrix)
    Y_pred = np.argmax(model.predict(X_input), axis=-1)
    
    predicted_emoji = emoji.emojize(emoji_dict[str(Y_pred[0])])
    
    return jsonify({
        'input_text': input_text,
        'predicted_emoji': predicted_emoji
    })

if __name__ == '__main__':
    app.run(debug=True)
