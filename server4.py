import socket
import numpy as np
import tensorflow as tf
import threading
from googletrans import Translator, LANGUAGES
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


IP = socket.gethostbyname(socket.gethostname()) #not assigned by the user/dynamic
PORT  = 4455
ADDR = (IP,PORT)
translator = Translator()
FORMAT = "utf-8" #ASCII type
SIZE = 1024

def handle_client(conn, addr):
    
    
    
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



    print(f"[NEW CONNECTION] {addr} connected.")


    data = conn.recv(SIZE).decode(FORMAT)

    print(f"[RECV] Text to be processed received.")


    test_padded = pad_sequences(data, maxlen=max_sequence_length)
    predicted_mood = np.argmax(model.predict(test_padded), axis=-1)

    conn.send(predicted_mood.text.encode(FORMAT))
    conn.close()
    print(f"[DISCONNECTED] {addr} disconnected")


def main():
    print("[STARTING] Server is starting.")
    print(f"[STARTING] Server is starting on {IP}:{PORT}")
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM) #TCP CONNECTION
    server.bind(ADDR)
    server.listen()
    print("[LISTENING] Server is listening.")

    while True:
        conn,addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


if __name__ == "__main__":
    main()
