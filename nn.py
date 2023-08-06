import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Data Preparation
sentences = ['I am happy', 'This is good', 'Feeling sad', 'Neutral sentence']
moods = ['positive', 'positive', 'negative', 'neutral']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

labels = np.array(moods)

# Step 2: Neural Network Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_sequence_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Step 3: Model Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)

# Step 4: Model Evaluation
test_sentence = ['I am feeling great']
test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_sequence, maxlen=max_sequence_length)
predicted_mood = np.argmax(model.predict(test_padded), axis=-1)

print('Predicted Mood:', predicted_mood)