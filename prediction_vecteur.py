import numpy as np
from numpy.linalg import norm
import os
import time
import re
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from random import randrange

'''
Modèle prédisant un vecteur. C'est le premier modèle que nous avons conçut.
Les raisons des changements qui ont aboutis au modèle "prediction_mot.py" sont
décrites dans notre rapport.
'''



# ***** Traitement du dataset *****

text = text = open("dataset_final.txt", 'rb').read().decode(encoding='utf-8')
text = text.lower()

# On obtient le texte découpé en liste de mots et ponctuation
splitted_text = re.findall(r"[\w']+|[.:,!?;()-]", text)
vocab = set(splitted_text)

# La dimension des vecteurs de mots (définie par la version de GloVe utilisée)
embedding_dim = 50

# Fonction pour ouvrir le fichier glove et retourner un dictionnaire d'association mot-vecteur
def read_glove_vector(glove_vec):
	with open(glove_vec, 'r', encoding='UTF-8') as f:
		word_to_vec_map = {}
		for line in f:
			w_line = line.split()
			curr_word = w_line[0]
			word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
	return word_to_vec_map


# Le dictionnaire d'association mot-vecteur
word_to_vec_map = read_glove_vector("glove.6B.50d.txt")

# Le modèle va prédire un vecteur qui ne sera jamais exactement similaire à un vecteur du fichier GloVe.
# On va donc devoir trouver le vecteur GloVe le plus proche du vecteur prédit pour trouver le mot prédit par
# le modèle. relevant_map stoque les associations mot:vecteur uniquement des mots présent dans note dataset.
# Cela permettra d'uniquement considérer ces vecteurs là pour trouver le vecteur le proche.
relevant_map = {}

for word in range(len(splitted_text)):
        curr_word = splitted_text[word]
        if curr_word == "$":
                if splitted_text[word-1] != '.' or splitted_text[word-1] != '!' or splitted_text[word-1] != '?':
                        curr_word = '.'
                else:
                        continue
        try:
                splitted_text[word] = word_to_vec_map[curr_word]
                relevant_map[curr_word] = word_to_vec_map[curr_word]
        except:
                try:
                        splitted_text[word] = word_to_vec_map[curr_word[:-1]]
                        relevant_map[curr_word[:-1]] = word_to_vec_map[curr_word[:-1]]
                except:
                        splitted_text[word] = word_to_vec_map["unknown"]
                        relevant_map["unknown"] = word_to_vec_map["unknown"]

relevant_words =  list(relevant_map.keys())
relevant_vectors = list(relevant_map.values())

# Fonction de similarité cosinus
def cosine_sim(a, b):
        return np.dot(a, b)/(norm(a)*norm(b))

# Méthode pour trouver le mot le plus proche dans notre vocabulaire. Notre modèle va générer des vecteurs qui ne seront jamais
# exactement similaires aux vecteurs de GloVe. Pour un vecteur prédit on va donc simplement trouver le mot de notre vocabulaire avec le vecteur
# le plus proche.
def vector2word(vector):
        # Pour accélérer les choses, on utilise des tableaux numpy
        full = np.full((len(relevant_vectors), embedding_dim), vector) 
        similarities = np.array(list(map(cosine_sim, relevant_vectors, full)))
        max_sim = np.max(similarities)
        index_max = np.where(similarities == max_sim )
        return relevant_words[index_max[0][0]]

def vectors2words(vectors):
        words = []
        for v in vectors:
                words.append(vector2word(v))
        return words

def words2vec(words):
        vecs = []
        for word in words:
                vecs.append(word_to_vec_map[word])
        return vecs

def average_similarity_in_text(vector):
        arr = np.array(splitted_text)
        full = np.full((len(splitted_text), embedding_dim), vector)
        similarities = np.array(list(map(cosine_sim, splitted_text, full)))
        return np.mean(similarities)

def average_vector():
         arr = np.array(splitted_text)
         avrg_vec = np.mean(arr, axis=0)
         return avrg_vec
                

                 
avg = average_vector()

print("Average vector: ", avg)
print("Vector associated with 'but':  ", word_to_vec_map["but"])
print("Cosine similarity between the average vector and the vector associated with 'but': ", cosine_sim(avg, word_to_vec_map["but"]))





# On va maintenant préparrer notre dataset pour l'entrainement
# Tout d'abord on découpe le dataset en séquences de seq_length+1 mots.
seq_length = 20
examples_per_epoch = len(splitted_text)//(seq_length+1)

tensor_text = tf.data.Dataset.from_tensor_slices(splitted_text)
sequences = tensor_text.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_vectors = sequence[:-1]
    target_vector = sequence[-1]
    return input_vectors, target_vector

dataset = sequences.map(split_input_target)


# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# ***** Contruction du modèle *****

# Nombre de neurones récurents
rnn_units = 100

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True))
model.add(tf.keras.layers.LSTM(rnn_units))
model.add(tf.keras.layers.Dense(embedding_dim , activation='linear'))

# ***** TRAINING *****

loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.AUTO, name='cosine_similarity')
model.compile(optimizer='adam', loss=loss)

EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS)


input_text = "Whatever you keep inside due to fear of ridicule or exposure will surface eventually. You will find however that".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)

# ====
input_text = "A dispute that has dragged on far too long must be resolved this weekend. Make an effort to see".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.append(yhat)

print(vectors2words(input_vectors_list))

input_text = "Whatever you keep inside due to fear of ridicule or exposure will surface eventually. You will find however that".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.append(yhat)

print(vectors2words(input_vectors_list))
# ====

input_text = "You".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)

input_text = "The Sun in your sign gives you the courage and the confidence you need to make the kind of changes".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)

input_text = "The energy at play today may make you initially more reserved and cautious in your attitude toward someone rather special".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)

input_text = "You would like to see those around you feeling good. Truly the most effective way of expressing this desire".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)

input_text = "The day 's planetary constellation is so sunny".lower()
input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
input_vectors_list = words2vec(input_words)

for iteration in range(20):
        input_vectors_tensor = tf.constant(np.array([input_vectors_list]))
        yhat = model.predict(input_vectors_tensor)[0]
        input_vectors_list.pop(0)
        input_words.append(vector2word(yhat))
        input_vectors_list.append(yhat)

print(input_words)


model.save('model_vector_pred_bis_shift.h5')


