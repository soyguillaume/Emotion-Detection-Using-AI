# Emotion Detection Using AI
Ce projet utilise l'intelligence artificielle pour détecter les émotions humaines en temps réel à partir de flux vidéo. Il repose sur un modèle de réseau neuronal pré-entrainé pour la détection d'émotions, intégré avec la détection de visages basée sur l'algorithme HaarCascade.

#### A vérifier avant de commencer
Pour exécuter le programme, s'assurer d'avoir les dépendances nécessaires (Keras, OpenCV, etc.) installées et un modèle Emotion_Detection.h5 disponible.


## Composants principaux
#### Classificateur de visages HaarCascade :
- Chargé depuis le fichier XML pré-entrainé.
- Utilisé pour détecter les visages dans chaque frame du flux vidéo.
#### Modèle de détection d'émotions (Emotion_Detection.h5) :

- Chargé à l'aide de Keras pour effectuer des prédictions sur les régions du visage détectées.
- Modèle pré-entrainé capable de prédire cinq émotions : Angry, Happy, Neutral, Sad, Surprise.
#### Capture vidéo (cv2.VideoCapture) :
- Initialise la capture vidéo à partir de la caméra par défaut.

## Boucle principale
- Boucle infinie traitant chaque frame du flux vidéo.
- Conversion de la frame en niveaux de gris pour simplifier le traitement.
- Utilisation du classificateur de visages pour détecter les visages dans la frame.
- Boucle sur chaque visage détecté, dessine un rectangle autour du visage et isole la région d'intérêt (ROI) du visage.
- Prépare la ROI pour être utilisée par le modèle de prédiction.
- Fait une prédiction d'émotion pour chaque ROI et affiche les résultats sur la frame.
- Affiche la frame avec les résultats de la détection d'émotion.
- Deux lignes vides dans la sortie pour aérer l'affichage.
- Sort de la boucle si la touche 'q' est pressée.

## Fin du programme
Libère les ressources de la caméra.
Ferme toutes les fenêtres d'affichage.

