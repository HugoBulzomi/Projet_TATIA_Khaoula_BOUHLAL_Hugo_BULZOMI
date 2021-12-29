import numpy as np
from numpy.linalg import norm
import os
import time
import re
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from random import randrange

'''
Modèle prédisant le procain mot encodé en one-hot. C'est le second modèle
que nous avons créé après l'échec de notre premier.

'''


# ***** Traitement du dataset *****

text = text = open("dataset_final.txt", 'rb').read().decode(encoding='utf-8')
text = text.lower()

# On obtient le texte découpé en liste de mots et ponctuation
splitted_text = re.findall(r"[\w']+|[.:,!?;()-]", text)
vocab = set(splitted_text)

# Chaque mot différent est assigné à un nombre entier qui servira lors de l'encodage one-hot.
# On a donc besoin de pouvoir faire la conversion mot->entier et entier->mot
ids_from_words = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
words_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_words.get_vocabulary(), invert=True, mask_token=None)
vocab_size = len(ids_from_words.get_vocabulary())

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


# Convertit notre dataset en ids et renvois un tensor qui permet d'itérer sur ces derniers
def splitted_dataset2ids(words):
        ids = ids_from_words(words)
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
        return ids_dataset


ids_dataset = splitted_dataset2ids(splitted_text)


# Le dictionnaire d'association mot-vecteur
word_to_vec_map = read_glove_vector("glove.6B.50d.txt")


for word in range(len(splitted_text)):
        curr_word = splitted_text[word]
        # On recherche le caractère de sépération et on ajoute un point si jamais il en manque
        if curr_word == "hugolemechant":
                if splitted_text[word-1] != '.' or splitted_text[word-1] != '!' or splitted_text[word-1] != '?':
                        curr_word = '.'
                else:
                        continue

        # Nous avons normalement enlevé les mots inconnus de notre dataset, mais il vaut mieux malgré tout vérifier qu'on ait bien
        # Un vecteur associé à chaque mot.
        try:
                # On essaie de trouver un vecteur pour le mot actuel
                splitted_text[word] = word_to_vec_map[curr_word]
        except:
                try:
                        # Si l'on y arrive pas, on essaie d'enlever la dernière lettre du mot (GloVe n'a parfois pas les versions pluriels de certains mots)
                        splitted_text[word] = word_to_vec_map[curr_word[:-1]]
 
                except:
                        # Dans le pire des cas, on utilise le vecteur du mot "unknown"
                        splitted_text[word] = word_to_vec_map["unknown"]






# On doit maintenant céér la matrice à donner à notre embedding layer pour qu'il convertisse les mots en vecteurs.
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word_id in range(vocab_size):
        word = words_from_ids(word_id).numpy().decode('UTF-8')

        try:
                embedding_vector = word_to_vec_map[word]
        except:
                embedding_vector = word_to_vec_map["unknown"]
  
        if embedding_vector is not None:
                embedding_matrix[word_id] = embedding_vector



# On va maintenant préparer notre dataset pour l'entrainement
# Tout d'abord on découpe le dataset en séquences de seq_length+1 mots.
seq_length = 20
examples_per_epoch = len(splitted_text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_ids = sequence[:-1]
    target_one_hot = tf.one_hot(sequence[-1], vocab_size, dtype=np.int32)
    return input_ids, target_one_hot

dataset = sequences.map(split_input_target)


# Batch size
BATCH_SIZE = 64


# Tensorflow met à notre disposition un objet Dataset pour faciliter l'entrainement
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
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False))
model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True))
model.add(tf.keras.layers.LSTM(rnn_units))
model.add(tf.keras.layers.Dense(rnn_units, activation='relu'))
model.add(tf.keras.layers.Dense(vocab_size , activation='softmax'))

# ***** TRAINING *****

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 100 

#model = tf.keras.models.load_model("HAAAAA_MODEL.h5")
#model.load_weights("HAAAAA_MODEL_WEIGHTS.h5")
for e in range(EPOCHS):
        history = model.fit(dataset, epochs=1)
        #model.save_weights("HAAAAA_MODEL_WEIGHTS.h5")

        # A partir de l'epoch 40, on génère du texte dans le fichier result.txt
        if e > 40:
                f = open("results.txt", "a")
                f.write("\n\n*** EPOCH {} ***\n".format(e))

                # Pour générer du texte, on voit que l'on doit prendre une phrase, la découper mots, convertir les mots en leurs id et les donner au modèle.
                input_text = "Whatever you keep inside due to fear of ridicule or exposure will surface eventually. You will find however that".lower()
                input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
                words = input_words.copy()
                print("words: ", words)
                input_ids = ids_from_words(input_words)

                # On itère ensuite sur un nombre arbitraire de prédiction pour générer le texte mot par mot
                for iteration in range(20):
                        yhat =  int(np.argmax(model.predict(tf.constant(np.array([input_ids])))[0]))
                        input_words.pop(0)
                        input_words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        input_ids = ids_from_words(input_words)

                print("words: ", words)
                f.write(' '.join(words))



                input_text = "A dispute that has dragged on far too long must be resolved this weekend. Make an effort to see".lower()
                input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
                words = input_words.copy()
                print("words: ", words)
                input_ids = ids_from_words(input_words)

                for iteration in range(20):
                        yhat =  int(np.argmax(model.predict(tf.constant(np.array([input_ids])))[0]))
                        input_words.pop(0)
                        input_words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        input_ids = ids_from_words(input_words)

                print("words: ", words)
                f.write('\n')
                f.write(' '.join(words))

                input_text = "let people into your life and they will take you good places !".lower()
                input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
                words = input_words.copy()
                print("words: ", words)
                input_ids = ids_from_words(input_words)

                for iteration in range(40):
                        yhat =  int(np.argmax(model.predict(tf.constant(np.array([input_ids])))[0]))
                        input_words.pop(0)
                        input_words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        input_ids = ids_from_words(input_words)

                print("words: ", words)
                f.write('\n')
                f.write(' '.join(words))


                input_text = "Today you may expect an unexpected".lower()
                input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
                words = input_words.copy()
                print("words: ", words)
                input_ids = ids_from_words(input_words)

                for iteration in range(20):
                        yhat =  int(np.argmax(model.predict(tf.constant(np.array([input_ids])))[0]))
                        input_words.pop(0)
                        input_words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        input_ids = ids_from_words(input_words)

                print("words: ", words)
                f.write('\n')
                f.write(' '.join(words))

                # On génère ici un texte entier de 600 mots.
                input_text = "your daily routine can be a great source of pleasure and discoveries this year if you can sit still long".lower()
                input_words = re.findall(r"[\w']+|[.:,!?;()-]", input_text)
                words = input_words.copy()
                print("words: ", words)
                input_ids = ids_from_words(input_words)

                for iteration in range(600):
                        yhat =  int(np.argmax(model.predict(tf.constant(np.array([input_ids])))[0]))
                        input_words.pop(0)
                        input_words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        words.append(words_from_ids(yhat).numpy().decode("UTF-8"))
                        input_ids = ids_from_words(input_words)

                print("words: ", words)
                f.write('\n')
                f.write(' '.join(words))

                f.close()


            


