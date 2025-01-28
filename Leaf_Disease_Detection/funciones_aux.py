import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
import random

###FUNCIONES PARA ENTRENAR Y EVALUAR LOS MODELOS

def evaluate_model(clasificador, X_test, y_test):
    """
    Realiza y muestra la evaluación de un clasificador en los datos de prueba
    
    Parameters:
    clasificador: modelo ya entrenado
    X_test (np.array): datos de test
    y_test (np.array): etiquetas de test
    
    """
    predictions = clasificador.predict(X_test)

    #Calculamos la precisión
    correct_predictions = np.sum(predictions == y_test)
    total_samples = len(y_test)
    print(f"\nImágenes correctamente clasificadas: {correct_predictions}/{total_samples}")
    print(f"Precisión total: {(correct_predictions / total_samples) * 100:.2f}%")
    
    #Mostramos la matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de Confusión")
    plt.show()

    #Mostramos un informe detallado con las medidas de la bondad del clasificador
    report = classification_report(y_test, predictions, target_names=["Negativo", "Positivo"])
    print("\nInforme detallado:\n", report)


def grid_search(training_data, classes):
    """
    Realiza una búsqueda en la cuadrícula para encontrar los mejores parámetros
    de un clasificador SVM utilizando scikit-learn.
    
    Parameters:
    training_data (np.array): datos de entrenamiento
    classes (np.array): etiquetas de entrenamiento
    
    Returns:
    GridSearchCV: modelo entrenado con los mejores parámetros
    """
    #Definimos los parámetros a explorar
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],  # 'linear' y 'rbf' son los equivalentes en sklearn
        'gamma': ['scale', 'auto']   # Para kernels no lineales como 'rbf'
    }
    
    #Definimos el clasificador
    svc = SVC()

    #Validaremos con validación cruzada estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Configuramos GridSearchCV
    grid_search = GridSearchCV(
        svc, 
        param_grid, 
        cv=skf,
        scoring='f1',  #Métrica para optimizar: F1-score
        verbose=3, #Mostrar más info
        #n_jobs=-1 #Usamos todos los nucleos
    )

    print("Ejecutando Grid Search...", flush=True)
    grid_search.fit(training_data, classes) #Buscamos los mejores hiperparámetros

    print("Mejores hiparámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor puntuación (F1): {grid_search.best_score_:.4f}")

    return grid_search


import pandas as pd


#Esta función la usaremos para comparar distintas combinaciones (HOG+SVM, LBP+SVM, etc)
def evaluate_svm(best_model, test_data, test_labels, descriptor_name):
    """
    Evalúa un modelo SVM ya entrenado (el mejor obtenido por GridSearchCV) usando los datos de test.

    Parameters:
        best_model: GridSearchCV, modelo entrenado con los mejores parámetros
        test_data: np.array, datos de prueba
        test_labels: np.array, etiquetas de prueba
        descriptor_name: str, nombre del descriptor usado (LBP o HOG)

    Returns:
        dict: diccionario con métricas de evaluación
    """
    # Generamos predicciones
    predictions = best_model.best_estimator_.predict(test_data)

    # Calculamos métricas
    accuracy = accuracy_score(test_labels, predictions)
    correct_predictions = np.sum(predictions == test_labels)
    total_samples = len(test_labels)
    
    
    report = classification_report(test_labels, predictions, target_names=["Negativo", "Positivo"], output_dict=True)
    cm = confusion_matrix(test_labels, predictions)
    
    # Mostramos resultados
    print(f"Resultados de SVM con {descriptor_name}:")
    
    print(f"\nImágenes correctamente clasificadas: {correct_predictions}/{total_samples}")
    print(f"Precisión total: {(correct_predictions / total_samples) * 100:.2f}%")
    
    print("Matriz de Confusión:")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de Confusión")
    plt.show()
    
    print("\nInforme de Clasificación:")
    print(classification_report(test_labels, predictions, target_names=["Negativo", "Positivo"]))
    #print(f"Accuracy: {accuracy:.2f}\n")

    # Devolvemos resultados como un diccionario
    return {
        "Descriptor": descriptor_name,
        "Accuracy": accuracy,
        "Precision Negativo": report["Negativo"]["precision"],
        "Recall Negativo": report["Negativo"]["recall"],
        "F1-Score Negativo": report["Negativo"]["f1-score"],
        "Precision Positivo": report["Positivo"]["precision"],
        "Recall Positivo": report["Positivo"]["recall"],
        "F1-Score Positivo": report["Positivo"]["f1-score"]
    }


### FUNCIONES PARA EL EJERCICIO DE 3 CLASES

def evaluate_model_multiple(clasificador, X_test, y_test, class_names):
    """
    Evalúa un modelo y muestra métricas de rendimiento para múltiples clases.

    Parameters:
        clasificador: Modelo entrenado (SVM u otro).
        X_test: np.array, datos de prueba.
        y_test: np.array, etiquetas de prueba.
        class_names: list, nombres de las clases.
    """
    # Generamos predicciones
    predictions = clasificador.predict(X_test)

    # Verificamos las clases presentes en las etiquetas y predicciones
    print("Clases en y_test:", np.unique(y_test))
    print("Clases en predicciones:", np.unique(predictions))

    # Calculamos la precisión
    correct_predictions = np.sum(predictions == y_test)
    total_samples = len(y_test)
    print(f"\nImágenes correctamente clasificadas: {correct_predictions}/{total_samples}")
    print(f"Precisión total: {(correct_predictions / total_samples) * 100:.2f}%")
    
    # Mostramos la matriz de confusión
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de Confusión")
    plt.show()

    # Mostramos un informe detallado con las medidas de la bondad del clasificador
    report = classification_report(y_test, predictions, target_names=class_names, labels=[0, 1, 2])
    print("\nInforme detallado:\n", report)


def grid_search_multiple(training_data, classes):
    """
    Realiza una búsqueda en la cuadrícula para encontrar los mejores parámetros
    de un clasificador SVM utilizando scikit-learn, soportando múltiples clases.
    
    Parameters:
    training_data (np.array): datos de entrenamiento
    classes (np.array): etiquetas de entrenamiento
    
    Returns:
    GridSearchCV: modelo entrenado con los mejores parámetros
    """
    #Definimos los parámetros a explorar
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']   # Para kernels no lineales como 'rbf'
    }
    
    #Definimos el clasificador
    svc = SVC(decision_function_shape='ovr')  #Configuración para múltiples clases

    #Validaremos con validación cruzada estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Configuramos GridSearchCV
    grid_search = GridSearchCV(
        svc, 
        param_grid, 
        cv=skf,
        scoring='f1_macro',  #Métrica para optimizar: F1-score macro para múltiples clases
        verbose=3, #Mostrar más info
    )

    print("Ejecutando Grid Search...", flush=True)
    grid_search.fit(training_data, classes) #Buscamos los mejores hiperparámetros

    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor puntuación (F1 Macro): {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_svm_multiple(best_model, test_data, test_labels, descriptor_name, class_names):
    """
    Evalúa un modelo SVM ya entrenado (el mejor obtenido por GridSearchCV) usando los datos de test.

    Parameters:
        best_model: GridSearchCV, modelo entrenado con los mejores parámetros
        test_data: np.array, datos de prueba
        test_labels: np.array, etiquetas de prueba
        descriptor_name: str, nombre del descriptor usado (LBP o HOG)
        class_names: list, nombres de las clases

    Returns:
        dict: diccionario con métricas de evaluación
    """
    # Generamos predicciones
    predictions = best_model.best_estimator_.predict(test_data)

    # Calculamos métricas
    accuracy = accuracy_score(test_labels, predictions)
    correct_predictions = np.sum(predictions == test_labels)
    total_samples = len(test_labels)

    report = classification_report(test_labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, predictions)

    # Mostramos resultados
    print(f"Resultados de SVM con {descriptor_name}:")
    print(f"\nImágenes correctamente clasificadas: {correct_predictions}/{total_samples}")
    print(f"Precisión total: {(correct_predictions / total_samples) * 100:.2f}%")

    print("Matriz de Confusión:")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de Confusión")
    plt.show()

    print("\nInforme de Clasificación:")
    print(classification_report(test_labels, predictions, target_names=class_names))

    # Devolvemos resultados como un diccionario
    metrics = {
        "Descriptor": descriptor_name,
        "Accuracy": accuracy
    }

    for class_name in class_names:
        metrics[f"Precision {class_name}"] = report[class_name]["precision"]
        metrics[f"Recall {class_name}"] = report[class_name]["recall"]
        metrics[f"F1-Score {class_name}"] = report[class_name]["f1-score"]

    return metrics