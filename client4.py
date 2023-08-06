import tkinter as tk
import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

PORT = 4455
SIZE = 1024
FORMAT = "utf-8"

root = tk.Tk()
root.title("the input to neural networks")

label_input = tk.Label(root, text="enter the text")
label_input.pack()

input_field = tk.Entry(root, width=50)
input_field.pack()

label_output = tk.Label(root, text="mood:")
label_output.pack()

output_field = tk.Text(root, height=5, width=50)
output_field.pack()

label_server_ip = tk.Label(root, text="Enter server IP address:")
label_server_ip.pack()

server_ip_field = tk.Entry(root, width=50)
server_ip_field.pack()

def mood_predict(ip_address):
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    ADDR = (ip_address, PORT)
    client.connect(ADDR)
    tokenizer = Tokenizer()
    input_text = input_field.get()
    test_sequence = tokenizer.texts_to_sequences(input_text)

    client.send(input_text.encode(FORMAT))
    mood = client.recv(SIZE).decode(FORMAT)
    print(f"[SERVER]: Text to be translated recieved")
    

    output_field.delete(1.0, tk.END)
    output_field.insert(tk.END, mood)


    client.close()


button_translate = tk.Button(root, text="Translate", command=lambda: mood_predict(server_ip_field.get()))
button_translate.pack()

root.mainloop()
