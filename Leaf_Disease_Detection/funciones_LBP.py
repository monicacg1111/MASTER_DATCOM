from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
import random

PATH_POSITIVE_TRAIN = "data_plantas/Train/Train/Rust/"
PATH_NEGATIVE_TRAIN = "data_plantas/Train/Train/Healthy/"
PATH_POSITIVE_TEST = "data_plantas/Test/Test/Rust/"
PATH_NEGATIVE_TEST = "data_plantas/Test/Test/Healthy/"
#EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop001002d_0.png" 
#EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST+"AnnotationsNeg_0.000000_00000002a_0.png"
IMAGE_EXTENSION = ".jpg"

### FUNCIONES PARA APLICAR EL DESCRIPTOR LBP
def load_training_data_LBP():
    """
    Lee las imágenes de entrenamiento (positivas y negativas) y calcula sus
    descriptores LBP para el entrenamiento.

    returns:
        np.array: numpy array con los descriptores de las imágenes leídas
        np.array: numpy array con las etiquetas de las imágenes leídas
    """ 
    training_data = []
    classes = []
    fixed_size = (128, 128)  #Redimensionamos las imágenes a tamaño fijo
    random.seed(42)  #Semilla para reproducibilidad

    #Configuración de LBP
    P = 8  # Número de vecinos
    R = 1   # Radio

    # Seleccionamos solo 100 imágenes positivas
    positive_files = [f for f in os.listdir(PATH_POSITIVE_TRAIN) if f.endswith(IMAGE_EXTENSION)]
    selected_positive_files = random.sample(positive_files, min(200, len(positive_files)))

    # Casos positivos
    for filename in selected_positive_files:
        filepath = os.path.join(PATH_POSITIVE_TRAIN, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error al leer la imagen {filepath}")
            continue
        img = cv2.resize(img, fixed_size)  #Redimensionamos la imagen
        img_gray = rgb2gray(img)  #Convertimos a escala de grises
        
        #Calculamos LBP y su histograma normalizado
        lbp = local_binary_pattern(img_gray, P, R)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  #Normalizamos el histograma
        
        training_data.append(hist)
        classes.append(1)

    print(f"Leídas {len(selected_positive_files)} imágenes de TRAIN -> positivas")

    # Seleccionamos solo 100 imágenes negativas
    negative_files = [f for f in os.listdir(PATH_NEGATIVE_TRAIN) if f.endswith(IMAGE_EXTENSION)]
    selected_negative_files = random.sample(negative_files, min(200, len(negative_files)))

    # Casos negativos
    for filename in selected_negative_files:
        filepath = os.path.join(PATH_NEGATIVE_TRAIN, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error al leer la imagen {filepath}")
            continue
        img = cv2.resize(img, fixed_size)  
        img_gray = rgb2gray(img) 

        lbp = local_binary_pattern(img_gray, P, R)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  

        training_data.append(hist) 
        classes.append(0)

    print(f"Leídas {len(selected_negative_files)} imágenes de TRAIN -> negativas")

    return np.array(training_data), np.array(classes)


def load_testing_data_LBP():
    """
    Lee las imágenes de test (positivas y negativas) y calcula sus
    descriptores LBP.

    returns:
        np.array: numpy array con los descriptores de las imágenes leídas
        np.array: numpy array con las etiquetas de las imágenes leídas
    """ 
    testing_data = []
    classes = []
    fixed_size = (128, 128)  #Redimensionamos las imágenes a tamaño fijo
    random.seed(42)  #Semilla para reproducibilidad

    #Configuración de LBP
    P = 8  #Número de vecinos
    R = 1  #Radio

    #Seleccionamos solo 50 imágenes positivas para test
    positive_files = [f for f in os.listdir(PATH_POSITIVE_TEST) if f.endswith(IMAGE_EXTENSION)]
    selected_positive_files = random.sample(positive_files, min(50, len(positive_files)))

    #Casos positivos
    for filename in selected_positive_files:
        filepath = os.path.join(PATH_POSITIVE_TEST, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error al leer la imagen {filepath}")
            continue
        img = cv2.resize(img, fixed_size)  #Redimensionamos la imagen
        img_gray = rgb2gray(img)  #Convertimos a escala de grises
        
        #Calculamos LBP y su histograma normalizado
        lbp = local_binary_pattern(img_gray, P, R)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  #Normalizamos el histograma
        
        testing_data.append(hist)
        classes.append(1)

    print(f"Leídas {len(selected_positive_files)} imágenes de TEST -> positivas")

    #Seleccionamos solo 50 imágenes negativas para test
    negative_files = [f for f in os.listdir(PATH_NEGATIVE_TEST) if f.endswith(IMAGE_EXTENSION)]
    selected_negative_files = random.sample(negative_files, min(50, len(negative_files)))

    #Casos negativos
    for filename in selected_negative_files:
        filepath = os.path.join(PATH_NEGATIVE_TEST, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error al leer la imagen {filepath}")
            continue
        img = cv2.resize(img, fixed_size)  #Redimensionamos la imagen
        img_gray = rgb2gray(img)  #Convertimos a escala de grises

        lbp = local_binary_pattern(img_gray, P, R)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  #Normalizamos el histograma

        testing_data.append(hist) 
        classes.append(0)

    print(f"Leídas {len(selected_negative_files)} imágenes de TEST -> negativas")

    return np.array(testing_data), np.array(classes)



PATH_HEALTHY_TRAIN = "data_plantas/Train/Train/Healthy/"
PATH_RUST_TRAIN = "data_plantas/Train/Train/Rust/"
PATH_POWDERY_TRAIN = "data_plantas/Train/Train/Powdery/"
PATH_HEALTHY_TEST = "data_plantas/Test/Test/Healthy/"
PATH_RUST_TEST = "data_plantas/Test/Test/Rust/"
PATH_POWDERY_TEST = "data_plantas/Test/Test/Powdery/"

def load_training_data_LBP_multiple():
    """
    Lee las imágenes de entrenamiento para múltiples clases (Healthy, Powdery, Rust) 
    y calcula sus descriptores LBP básico.

    returns:
        np.array: numpy array con los descriptores de las imágenes leídas
        np.array: numpy array con las etiquetas de las imágenes leídas
    """ 
    training_data = []
    classes = []
    fixed_size = (128, 128) #Redimensionamos las imágenes a tamaño fijo
    random.seed(42) #Semilla para reproducibilidad

    #Configuración de LBP
    P = 8 #Número de vecinos
    R = 1 #Radio

    #Diccionario con las rutas y etiquetas de cada clase
    class_paths = {
        0: PATH_HEALTHY_TRAIN,
        1: PATH_POWDERY_TRAIN,
        2: PATH_RUST_TRAIN
    }

    #Procesar cada clase
    for label, path in class_paths.items():
        files = [f for f in os.listdir(path) if f.endswith(IMAGE_EXTENSION)]
        selected_files = random.sample(files, min(200, len(files)))

        for filename in selected_files:
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Error al leer la imagen {filepath}")
                continue
            img = cv2.resize(img, fixed_size) #Redimensionamos la imagen
            img_gray = rgb2gray(img) #Convertimos a escala de grises
            
            #Calculamos LBP básico y su histograma
            lbp = local_binary_pattern(img_gray, P, R)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6) #Normalizamos el histograma

            training_data.append(hist) #Añadimos el histograma normalizado
            classes.append(label)

        print(f"Leídas {len(selected_files)} imágenes de TRAIN -> Clase {label}")

    return np.array(training_data), np.array(classes)


def load_testing_data_LBP_multiple():
    """
    Lee las imágenes de prueba para múltiples clases (Healthy, Powdery, Rust) 
    y calcula sus descriptores LBP básico.

    returns:
        np.array: numpy array con los descriptores de las imágenes leídas
        np.array: numpy array con las etiquetas de las imágenes leídas
    """ 
    testing_data = []
    classes = []
    fixed_size = (128, 128) #Redimensionamos las imágenes a tamaño fijo
    random.seed(42) #Semilla para reproducibilidad

    #Configuración de LBP
    P = 8 #Número de vecinos
    R = 1 #Radio

    #Diccionario con las rutas y etiquetas de cada clase
    class_paths = {
        0: PATH_HEALTHY_TEST,
        1: PATH_POWDERY_TEST,
        2: PATH_RUST_TEST
    }

    #Procesar cada clase
    for label, path in class_paths.items():
        files = [f for f in os.listdir(path) if f.endswith(IMAGE_EXTENSION)]
        selected_files = random.sample(files, min(50, len(files))) #Seleccionamos solo 50 imágenes para prueba

        for filename in selected_files:
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Error al leer la imagen {filepath}")
                continue
            img = cv2.resize(img, fixed_size) #Redimensionamos la imagen
            img_gray = rgb2gray(img) #Convertimos a escala de grises

            #Calculamos LBP básico y su histograma
            lbp = local_binary_pattern(img_gray, P, R)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6) #Normalizamos el histograma

            testing_data.append(hist) #Añadimos el histograma normalizado
            classes.append(label)

        print(f"Leídas {len(selected_files)} imágenes de TEST -> Clase {label}")

    return np.array(testing_data), np.array(classes)