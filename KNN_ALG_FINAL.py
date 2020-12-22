"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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

algoritmo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
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
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()

# Creamos el modelo
cv = KFold(n_splits=10)  # Numero deseado de "folds" que haremos
accuracies = list()
nn = range(1, 20)
# Parámetros
"""
n_neighbors : int, default=5
    Number of neighbors to use by default for kneighbors queries.

weights : {‘uniform’, ‘distance’} or callable, default=’uniform’
    weight function used in prediction. Possible values:

    ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
    ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of 
    a query point will have a greater influence than neighbors which are further away.
    [callable] : a user-defined function which accepts an array of distances, and returns an array 
    of the same shape containing the weights.

algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
    Algorithm used to compute the nearest neighbors:
    ‘ball_tree’ will use BallTree
    ‘kd_tree’ will use KDTree
    ‘brute’ will use a brute-force search.
    ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
    Note: fitting on sparse input will override the setting of this parameter, using brute force.

leaf_size : int, default=30
    Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, 
    as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

p : int, default=2
    Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), 
    and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric : str or callable, default=’minkowski’
    the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the 
    standard Euclidean metric. See the documentation of DistanceMetric for a list of available metrics. 
    If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. 
    X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

n_jobs : int, default=None
    The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. 
    -1 means using all processors. See Glossary for more details. Doesn’t affect fit method.	
"""
# Testearemos cada vecindad de 1 a 9
for n_neighbors in nn:
    fold_accuracy = []
    knn = KNeighborsClassifier(n_neighbors, metric='minkowski', p=2)

    for train_fold, valid_fold in cv.split(data):
        f_train = data.loc[train_fold]
        f_valid = data.loc[valid_fold]

        model = knn.fit(X=f_train.drop(['hora_ideal'], axis=1),
                        y=f_train["hora_ideal"])
        valid_acc = model.score(X=f_valid.drop(['hora_ideal'], axis=1),
                                y=f_valid["hora_ideal"])  # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"Vecinos": nn, "Precision": accuracies})
df = df[["Vecinos", "Precision"]]
print(df.to_string(index=False))
print('\n\nModelo creado satisfactoriamente!')
plt.style.use("ggplot")
plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(nn, accuracies, label='Precisión')
plt.xlabel('Vecinos (K)')
plt.ylabel('Precisión')
plt.title("Precisión por cantidad de vecinos")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.xticks([0, 5, 10, 15, 20])
plt.savefig('KNN_grafico_prec.png', dpi=300)
plt.show()

best_neighbors = 0
best_neighbors_pst = 0
for i in range(len(accuracies)):
    if accuracies[i] > best_neighbors:
        best_neighbors = accuracies[i]
        best_neighbors_pst = i + 1
print("El mejor es: ", best_neighbors, " en ", best_neighbors_pst)

# Seleccionamos las columnas
X = data.drop(['hora_ideal'], axis=1)
# Se deine los datos correspondientes a la etiqueta
y = data["hora_ideal"]

# # IMPLEMENTACIÓN DE ÁRBOLES DE DECISIÓN CLASIFICACIÓN ##
# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Defino el algoritmo a utilizar

algoritmo = KNeighborsClassifier(n_neighbors=best_neighbors_pst, metric='minkowski', p=2)
# Entreno el modelo
algoritmo.fit(X_train, y_train)
# serialize our model and save it in the file area_model.pickle
print("Model trained. Saving model to area_model.pickle")
with open("knn_algorithm.pickle", "wb") as file:
    pickle.dump(algoritmo, file)
print("Modelo guardado")
# Realiso una predicción
y_pred = algoritmo.predict(X_test)
# Verifico la matriz de confusión
print('Precisión de K-NN "Clasificación"  en el conjunto de entrenamiento: {:.2f}'
      .format(algoritmo.score(X_train, y_train)))
print('Precisión del K-NN "Clasificación" en el conjunto de prueba: {:.2f}'
      .format(algoritmo.score(X_test, y_test)))
matriz = confusion_matrix(y_test, y_pred)
target_names = ['Horario A', 'Horario B', 'Horario C', 'Horario D', 'Horario E', 'Horario F']
report_class = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print("MATRIZ DE CONFUSIÓN:")
print(matriz)
print("REPORTE DE CLASIFICACIÓN")
print(report_class)
# Calculo la precisión del modelo
precision = precision_score(y_test, y_pred, average='micro')
print("Precisión del modelo:")
print(precision)
# Validación cruzada
print("Validadción cruzada")
scores = cross_val_score(algoritmo, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def knn_comparison(dataS, k, mt, pp, wgts):
    Xs = dataS.drop(['hora_ideal'], axis=1).values
    # Se deFine los datos correspondientes a la etiqueta
    ys = dataS["hora_ideal"].values
    clf = KNeighborsClassifier(n_neighbors=k, metric=mt, p=pp, weights=wgts)
    pca = PCA(n_components=2)
    X_train2 = pca.fit_transform(Xs)
    clf.fit(X_train2, ys)
    # Plotting decision region
    plt.figure(figsize=(8, 5), dpi=300)
    plot_decision_regions(X_train2, ys, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('Características')
    plt.ylabel('Objetivo')
    plt.title("K-NN :Límite de la región de decisión con K=" + str(k))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig('KNN_Limit.png', dpi=300)
    plt.show()


metric = 'minkowski'
p = 2
weights = 'distance'
knn_comparison(data, best_neighbors_pst, metric, p, weights)
