import funciones_aux, funciones_HOG, funciones_LBP, funciones_LBPu
import pandas as pd


#Función para el menú principal
def main():
    print("\nSeleccione una opción:")
    print("Opción 1: Clasificación con 2 clases (Healthy y Rust)")
    print("Opción 2: Clasificación con 3 clases (Healthy, Rust y Powdery)")
    opcion = input("Ingrese la opción deseada: ")

    if opcion == "1":
        ejecutar_clasificacion_2_clases()
    elif opcion == "2":
        ejecutar_clasificacion_3_clases()
    else:
        print("Opción no válida. Saliendo del programa.")

# Ejecución para 2 clases
def ejecutar_clasificacion_2_clases():
    print("\n*** CLASIFICACIÓN CON 2 CLASES ***")

    # Descriptor HOG
    print("Leyendo imágenes con descriptor HOG para 2 clases...")
    X_train_HOG, y_train_HOG = funciones_HOG.load_training_data_HOG()
    X_test_HOG, y_test_HOG = funciones_HOG.load_testing_data_HOG()

    mejor_modelo_HOG = funciones_aux.grid_search(X_train_HOG, y_train_HOG)
    funciones_aux.evaluate_model(mejor_modelo_HOG, X_test_HOG, y_test_HOG)

    # Descriptor LBP
    print("Leyendo imágenes con descriptor LBP para 2 clases...")
    X_train_LBP, y_train_LBP = funciones_LBP.load_training_data_LBP()
    X_test_LBP, y_test_LBP = funciones_LBP.load_testing_data_LBP()

    mejor_modelo_LBP = funciones_aux.grid_search(X_train_LBP, y_train_LBP)
    funciones_aux.evaluate_model(mejor_modelo_LBP, X_test_LBP, y_test_LBP)

    # Comparación HOG y LBP
    print("Evaluando los mejores modelos para SVM+HOG y SVM+LBP...")
    results_HOG = funciones_aux.evaluate_svm(mejor_modelo_HOG, X_test_HOG, y_test_HOG, "HOG")
    results_LBP = funciones_aux.evaluate_svm(mejor_modelo_LBP, X_test_LBP, y_test_LBP, "LBP")

    results_df = pd.DataFrame([results_HOG, results_LBP])
    print("Comparativa de resultados:")
    print(results_df)

    # Descriptor LBP uniforme
    print("Leyendo imágenes con descriptor LBP uniforme para 2 clases...")
    X_train_LBP_uniform, y_train_LBP_uniform = funciones_LBPu.load_training_data_LBP_uniform()
    X_test_LBP_uniform, y_test_LBP_uniform = funciones_LBPu.load_testing_data_LBP_uniform()

    mejor_modelo_LBP_uniform = funciones_aux.grid_search(X_train_LBP_uniform, y_train_LBP_uniform)
    results_LBP_uniform = funciones_aux.evaluate_svm(mejor_modelo_LBP_uniform, X_test_LBP_uniform, y_test_LBP_uniform, "LBP-Uniform")

    # Comparación final
    results_df = pd.DataFrame([results_HOG, results_LBP, results_LBP_uniform])
    print("Comparativa de resultados:")
    print(results_df)

# Ejecución para 3 clases
def ejecutar_clasificacion_3_clases():
    print("\n*** CLASIFICACIÓN CON 3 CLASES ***")

    # Descriptor HOG
    print("Leyendo imágenes con descriptor HOG para 3 clases...")
    X_train_HOG, y_train_HOG = funciones_HOG.load_training_data_HOG_multiple()
    X_test_HOG, y_test_HOG = funciones_HOG.load_testing_data_HOG_multiple()

    mejor_modelo_HOG = funciones_aux.grid_search_multiple(X_train_HOG, y_train_HOG)
    funciones_aux.evaluate_model_multiple(mejor_modelo_HOG, X_test_HOG, y_test_HOG, ["Healthy", "Rust", "Powdery"])

    # Descriptor LBP
    print("Leyendo imágenes con descriptor LBP para 3 clases...")
    X_train_LBP, y_train_LBP = funciones_LBP.load_training_data_LBP_multiple()
    X_test_LBP, y_test_LBP = funciones_LBP.load_testing_data_LBP_multiple()

    mejor_modelo_LBP = funciones_aux.grid_search_multiple(X_train_LBP, y_train_LBP)
    funciones_aux.evaluate_model_multiple(mejor_modelo_LBP, X_test_LBP, y_test_LBP, ["Healthy", "Rust", "Powdery"])

    # Comparación HOG y LBP
    print("Evaluando los mejores modelos para SVM+HOG y SVM+LBP...")
    results_HOG = funciones_aux.evaluate_svm_multiple(mejor_modelo_HOG, X_test_HOG, y_test_HOG, "HOG", ["Healthy", "Rust", "Powdery"])
    
    results_LBP = funciones_aux.evaluate_svm_multiple(mejor_modelo_LBP, X_test_LBP, y_test_LBP, "LBP", ["Healthy", "Rust", "Powdery"])
    #results_HOG=results_LBP

    results_df = pd.DataFrame([results_HOG, results_LBP])
    print("Comparativa de resultados:")
    print(results_df)

    # Descriptor LBP uniforme
    print("Leyendo imágenes con descriptor LBP uniforme para 3 clases...")
    X_train_LBP_uniform, y_train_LBP_uniform = funciones_LBPu.load_training_data_LBP_uniform_multiple()
    X_test_LBP_uniform, y_test_LBP_uniform = funciones_LBPu.load_testing_data_LBP_uniform_multiple()

    mejor_modelo_LBP_uniform = funciones_aux.grid_search(X_train_LBP_uniform, y_train_LBP_uniform)
    results_LBP_uniform = funciones_aux.evaluate_svm_multiple(mejor_modelo_LBP_uniform, X_test_LBP_uniform, y_test_LBP_uniform, "LBP-Uniform", ["Healthy", "Rust", "Powdery"])

    # Comparación final
    results_df = pd.DataFrame([results_HOG, results_LBP, results_LBP_uniform])
    print("Comparativa de resultados:")
    print(results_df)


if __name__ == "__main__":
    main()