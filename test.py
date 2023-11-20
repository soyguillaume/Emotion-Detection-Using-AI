# PyPower Projects
# Emotion Detection Using AI



from keras.models import load_model #pour charger un modèle sauvegardé et déjà entrainé  (ici : Emotion_Detection.h5)
from time import sleep
from keras.preprocessing.image import img_to_array # convertit une image en un tableau numpy
from keras.preprocessing import image #module image de Keras, qui contient des utilitaires pour le prétraitement des images, comme la chargement et le redimensionnement
import cv2 #OpenCV (cv2), pour le traitement d'image en vision par ordinateur : capture vidéo, la manipulation d'images, la détection de visages
import numpy as np #pour manipuler des tableaux et des matrices, ce qui est fréquemment nécessaire lors du traitement d'images dans le contexte du machine learning.

#? motion_Detection.h5 = modèle de réseau neuronal construit avec une bibliothèque Keras ou TensorFlow
    #? chargé avec load_model  effectue des prédictions sur de nouvelles images pour déterminer les émotions présentes.
        # ?Prédire les émotions

#?haarcascade_frontalface_default.xml = algorithme de détection de visages basé sur des caractéristiques visuelles
        # ?Détecter les visages


#------------------- CODE -----------------------------

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# Charge le classificateur de visages HaarCascade pré-entrainé qui sera utilisé pour détecter les visages

classifier = load_model('./Emotion_Detection.h5')
# Charge le modèle de réseau neuronal pré-entrainé pour la détection d'émotions.

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
# Définit les étiquettes des classes d'émotions possibles que le modèle peut prédire.

cap = cv2.VideoCapture(0)
# Initialise la capture vidéo à partir de la caméra par défaut (index 0 = 1ère camréra trouvé) ici j'ai que ma webcam

while True:
    # Boucle infinie pour traiter chaque frame du flux vidéo.

    ret, frame = cap.read()
    # Lire une frame du flux vidéo.

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #RVB = BGR to GRAY
    # Convertit la frame en niveaux de gris pour simplifier le traitement.
        #//Les niveaux de gris réduisent la complexité (1 canal nuance de gris vs 3 en RVB) pour un traitement plus rapide

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # Utilise le classificateur de visages pour détecter les visages dans la frame.

    for (x, y, w, h) in faces:
        # Boucle sur chaque visage détecté.

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Dessine un rectangle autour du visage détecté.
            #//cv2.rectangle: Fonction OpenCV pour dessiner un rectangle.
                #//frame: Image sur laquelle le rectangle est dessiné.
                    #//(x, y): Coin supérieur gauche du rectangle (coordonnées du visage détecté).
                        #//(x+w, y+h): Coin inférieur droit du rectangle (coordonnées du visage détecté).
                            #//(255, 0, 0): Couleur du rectangle (bleu ici).
                                #//2: Épaisseur du trait du rectangle.

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # Isole la région d'intérêt (ROI) du visage, la redimensionne pour correspondre aux attentes du modèle.

        if np.sum([roi_gray]) != 0:
            # Vérifie si la Region Of Interest (ROI) contient des pixels.
            
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # Prépare la ROI pour être utilisée par le modèle de prédiction.

            preds = classifier.predict(roi)[0]
            # Fait une prédiction d'émotion pour la ROI.

            print("\nprediction = ", preds)
            label = class_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # Affiche le résultat de la prédiction sur la frame.

        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # Affiche un message si aucun visage n'est trouvé dans la frame.

        print("\n\n")
        #// "\n\n":  produit deux lignes vides dans la sortie pour aérer dans la console
    
    cv2.imshow('Emotion Detector', frame)
    # Affiche la frame avec les résultats de la détection d'émotion.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Sort de la boucle si la touche 'q' est pressée.

cap.release()
cv2.destroyAllWindows()
# Libère les ressources de la caméra et ferme toutes les fenêtres d'affichage.
