# Importe les bibliothèques nécessaires pour construire et entraîner un modèle de classification d'émotions.

from keras.applications import MobileNet
from keras.models import Sequential, Model 
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Définit la taille des images d'entrée pour le modèle MobileNet.
img_rows, img_cols = 224, 224

# Charge le modèle MobileNet pré-entrainé sans la couche fully connected (top layer).
MobileNet = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Gèle les quatre dernières couches du modèle MobileNet.
for layer in MobileNet.layers:
    layer.trainable = True

# Affiche les informations sur chaque couche du modèle MobileNet.
for (i, layer) in enumerate(MobileNet.layers):
    print(str(i), layer.__class__.__name__, layer.trainable)

# Fonction qui crée la partie supérieure du modèle (fully connected layers).
def addTopModelMobileNet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

# Nombre de classes (émotions) dans le problème de classification.
num_classes = 5

# Crée la partie supérieure du modèle en utilisant la fonction définie précédemment.
FC_Head = addTopModelMobileNet(MobileNet, num_classes)

# Crée le modèle complet en spécifiant les entrées et les sorties.
model = Model(inputs=MobileNet.input, outputs=FC_Head)

# Affiche un résumé du modèle, montrant l'architecture et le nombre de paramètres.
print(model.summary())

# Définit les répertoires des données d'entraînement et de validation.
train_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/train'
validation_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'

# Configuration de l'augmentation des données pour l'entraînement.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Configuration de l'augmentation des données pour la validation.
validation_datagen = ImageDataGenerator(rescale=1./255)

# Taille des lots d'images pour l'entraînement et la validation.
batch_size = 32

# Générateur d'images à partir du répertoire d'entraînement.
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Générateur d'images à partir du répertoire de validation.
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Importe les optimiseurs et les rappels nécessaires.
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration des rappels pour sauvegarder le modèle, arrêter l'entraînement prématurément et ajuster le taux d'apprentissage.
checkpoint = ModelCheckpoint(
    'emotion_face_mobilNet.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc',
    patience=5,
    verbose=1,
    factor=0.2,
    min_lr=0.0001
)

# Liste des rappels à utiliser pendant l'entraînement.
callbacks = [earlystop, checkpoint, learning_rate_reduction]

# Compilation du modèle en spécifiant la fonction de perte, l'optimiseur et les métriques.
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

# Nombre total d'échantillons d'entraînement et de validation.
nb_train_samples = 24176
nb_validation_samples = 3006

# Nombre d'époques d'entraînement.
epochs = 25

#
