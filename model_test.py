import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Carica il modello addestrato
model = load_model('cifar10_cnn_model.h5')

# Classi CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Funzione per fare predizioni
def predict_image(image_path):
    # Carica e processa l'immagine
    img = load_img(image_path, target_size=(32, 32))  # Adatta la dimensione alle richieste del modello
    img_array = img_to_array(img) / 255.0  # Normalizza
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi una dimensione batch

    # Predizione
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    # Stampa risultato
    print(f"Predicted Class: {class_names[class_idx]} with confidence {confidence:.2f}")
    return class_names[class_idx], confidence

# Esegui predizioni su un'immagine
image_path = '/Users/francescomorandi/Downloads/cifar-10/test/325.png'  # Cambia con il percorso dell'immagine
predict_image(image_path)
