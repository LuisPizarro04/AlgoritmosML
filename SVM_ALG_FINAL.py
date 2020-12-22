"""
Import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()
dataset = pd.DataFrame(data)
# Seleccionamos las columnas
X = dataset.drop(['hora_ideal'], axis=1)
# Se deine los datos correspondientes a la etiqueta
y = dataset["hora_ideal"]

# # IMPLEMENTACIÓN DE ÁRBOLES DE DECISIÓN CLASIFICACIÓN ##
# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Defino el algoritmo a utilizar

algoritmo = SVC(kernel='linear', C=0.5)
# Entreno el modelo
algoritmo.fit(X_train, y_train)

# Realiso una predicción
y_pred = algoritmo.predict(X_test)

# Verifico la matriz de confusión
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(algoritmo.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(algoritmo.score(X_test, y_test)))

matriz = confusion_matrix(y_test, y_pred)
report_class = classification_report(y_test, y_pred)
print("Matriz de confsión:")
print(matriz)
print("Reporte de clasificación")
print(report_class)
# Calculo la precisión del modelo
precision = precision_score(y_test, y_pred, average='micro')
print("Precisión del modelo:")
print(precision)
"""
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()

# Creamos el modelo
cv = KFold(n_splits=30)  # Numero deseado de "folds" que haremos
accuracies = list()

# Parametros
"""
C : float, default=1.0
    Regularization parameter. The strength of the regularization is inversely proportional to C. 
    Must be strictly positive. The penalty is a squared l2 penalty.

kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
    ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute 
    the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

degree : int, default=3
    Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

gamma : {‘scale’, ‘auto’} or float, default=’scale’
    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
    if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
    if ‘auto’, uses 1 / n_features.
    Changed in version 0.22: The default value of gamma changed from ‘auto’ to ‘scale’.

coef0 : float, default=0.0
    Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

shrinking : bool, default=True
    Whether to use the shrinking heuristic. See the User Guide.

probability : bool, default=False
    Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method 
    as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict. 
    Read more in the User Guide.

tol : float, default=1e-3
    Tolerance for stopping criterion.

cache_size : float, default=200
    Specify the size of the kernel cache (in MB).

class_weight : dict or ‘balanced’, default=None
    Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight 
    one. 
    The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class 
    frequencies in the input data as n_samples / (n_classes * np.bincount(y))

verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, 
    if enabled, may not work properly in a multithreaded context.

max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.

decision_function_shape : {‘ovo’, ‘ovr’}, default=’ovr’
    Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, 
    or the original one-vs-one (‘ovo’) decision function of libsvm which has shape 
    (n_samples, n_classes * (n_classes - 1) / 2). 
    However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary 
    classification.
    Changed in version 0.19: decision_function_shape is ‘ovr’ by default.
    New in version 0.17: decision_function_shape=’ovr’ is recommended.
    Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.

break_ties : bool, default=False
    If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the
    confidence values  of decision_function; otherwise the first class among the tied classes is returned. 
    Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.
    New in version 0.22.

random_state : int or RandomState instance, default=None
    Controls the pseudo random number generation for shuffling the data for probability estimates. 
    Ignored when probability is False. Pass an int for reproducible output across multiple function calls. 
    See Glossary.
"""

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
cs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for k_v in kernels:
    for c_v in cs:
        fold_accuracy = []
        svm = SVC(kernel=k_v, C=c_v)

        for train_fold, valid_fold in cv.split(data):
            f_train = data.loc[train_fold]
            f_valid = data.loc[valid_fold]

            model = svm.fit(X=f_train.drop(['hora_ideal'], axis=1),
                            y=f_train["hora_ideal"])
            valid_acc = model.score(X=f_valid.drop(['hora_ideal'], axis=1),
                                    y=f_valid["hora_ideal"])  # calculamos la precision con el segmento de validacion
            fold_accuracy.append(valid_acc)

        avg = sum(fold_accuracy) / len(fold_accuracy)
        accuracies.append(avg)
    # Mostramos los resultados obtenidos
    df = pd.DataFrame({"Kernel": k_v, "C": cs, "Precision": accuracies})
    df = df[["Kernel", "C", "Precision"]]
    print(df.to_string(index=False))
    print('\n\nModelo creado satisfactoriamente!')
    accuracies.clear()


def svm_comparison(datas):
    X = datas.drop(['hora_ideal'], axis=1).values
    # Se deFine los datos correspondientes a la etiqueta
    y = datas["hora_ideal"].values
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X)
    X_train2, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2)
    clf = SVC(kernel='sigmoid', C=0.5)
    clf.fit(X_train2, y_train)
    y_pred = clf.predict(X_test)
    # Plotting decision region
    plt.figure(figsize=(8, 5), dpi=300)
    plot_decision_regions(X_train2, y_train, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('Características')
    plt.ylabel('Objetivo')
    plt.title("SVM:Límite de la región de decisión con 'kernel'= sigmoid y 'C' =0.5")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('SVM_Limit.png', dpi=300)

    matriz = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión")
    print(matriz)

    plt.show()


X = data.drop(['hora_ideal'], axis=1)
y = data["hora_ideal"]
algoritmo = SVC(kernel='sigmoid', C=0.5)
# Validación cruzada
print("Validadción cruzada")
scores = cross_val_score(algoritmo, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

svm_comparison(data)
