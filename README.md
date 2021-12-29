# Projet_TATIA_Khaoula_BOUHLAL_Hugo_BULZOMI
Le dépot github de notre projet TATIA: Génération de texte par prédiction successive du prochain mot d’un texte `a l’aide d’un réseau de neurones récurrents

Attention: nous n'avons pas pu inclure le fichier glove qe nous avons utilisé car github le considère comme trop lourd. Nous l'avons téléchargé à l'adresse: https://nlp.stanford.edu/projects/glove/ (version Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download)) 

Le dataset:
dataset_final.txt est le dataset que nous avons utilisé.

Les résultats:
results.txt contiens le texte généré par le modèle de prediction_mot.py. 

Le rapport:
Le rapport final est contenu dans le pdf Rapport_TATIA_Khaoula_BOUHLAL_Hugo_BULZOMI.pdf

Les modèles présents ici:
Le fichier prediction_caractere.py est le modèle que nous utilisons pour comparer nos résultats. Il est repris de la documentation Tensorflow: https://www.tensorflow.org/text/tutorials/text_generation

Le fichier prediction_mot.py est quant à lui le second modèle que nous avons créé, comme expliqué dans notre rapport.

Enfin le fichier prediction_vecteur.py est le premier modèle que nous avons créé.

Nous nous excusons pour le retard de soumission de notre projet. Nous avons cru que l'heure de soumission à 00h00 était pour le jour suivant...
