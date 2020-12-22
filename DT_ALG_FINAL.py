from sklearn import tree
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pydot

plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('ggplot')

data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()

cv = KFold(n_splits=35)  # Numero deseado de "folds" que haremos
accuracies = list()
depth_range = range(1, 20)
# Parámetros
"""
Arbol de decisión.
criterion : string, optional (default=”gini”)
    La función para medir la calidad de una división. Los criterios admitidos son "gini" 
    para la impureza de Gini y "entropía" para la ganancia de información.
splitter : string, optional (default=”best”)
    La estrategia utilizada para elegir la división en cada nodo. Las estrategias admitidas son 
    "mejores" para elegir la mejor división y "aleatorias" para elegir la mejor división aleatoria.
max_features : int, float, string or None, optional (default=None)
    La cantidad de características que se deben considerar al buscar la mejor división:
    Si es int, entonces considere las características de max_features en cada división.
    Si es flotante, max_features es un porcentaje y las características--
    --int (max_features * n_features) se consideran en cada división.
    Si es "auto", entonces max_features = sqrt (n_features) .
    Si es "sqrt", entonces max_features = sqrt (n_features) .
    Si es "log2", entonces max_features = log2 (n_features) .
    Si es Ninguno, entonces max_features = n_features .
    Nota: la búsqueda de una división no se detiene hasta que se encuentra al menos una 
    partición válida de las muestras de nodos, incluso si requiere inspeccionar de manera 
    efectiva más de las características de max_features .
max_depth : int or None, optional (default=None)
    La profundidad máxima del árbol. Si es None, los nodos se expanden hasta que todas 
    las hojas sean puras o hasta que todas las hojas contengan menos de min_samples_split 
    muestras. Se ignora si max_samples_leaf no es None.
min_samples_split : int, optional (default=2)
    El número mínimo de muestras necesarias para dividir un nodo interno.
min_samples_leaf : int, optional (default=1)
    El número mínimo de muestras necesarias para estar en un nodo hoja.
max_leaf_nodes : int or None, optional (default=None)
    Haga crecer un árbol con max_leaf_nodes de la mejor manera primero. Los mejores nodos
    se definen como una reducción relativa de la impureza. Si es Ninguno, entonces un 
    número ilimitado de nodos hoja. Si no es None , se ignorará max_depth.
random_state : int, RandomState instance or None, optional (default=None)
    Si es int, random_state es la semilla usada por el generador de números aleatorios; Si es 
    una instancia de RandomState, random_state es el generador de números aleatorios; Si es 
    None, el generador de números aleatorios es la instancia de RandomState utilizada 
    por np.random.
"""
# Testearemos la profundidad de 1 hasta 5
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='gini',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth=depth,
                                             class_weight={1: 3.5})
    for train_fold, valid_fold in cv.split(data):
        f_train = data.loc[train_fold]
        f_valid = data.loc[valid_fold]

        model = tree_model.fit(X=f_train.drop(['hora_ideal'], axis=1),
                               y=f_train["hora_ideal"])
        valid_acc = model.score(X=f_valid.drop(['hora_ideal'], axis=1),
                                y=f_valid["hora_ideal"])  # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"Max Prof": depth_range, "Precision": accuracies})
df = df[["Max Prof", "Precision"]]
print(df.to_string(index=False))
plt.figure()
plt.xlabel('Depth')
plt.ylabel('accuracy')
plt.scatter(depth_range, accuracies)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

best_depth = 0
best_depth_pst = 0
for i in range(len(accuracies)):
    if accuracies[i] > best_depth:
        best_depth = accuracies[i]
        best_depth_pst = i + 1
print("El mejor es: ", best_depth, " en ", best_depth_pst)

# Seleccionamos las columnas
X = data.drop(['hora_ideal'], axis=1)
# Se deine los datos correspondientes a la etiqueta
y = data["hora_ideal"]

# # IMPLEMENTACIÓN DE ÁRBOLES DE DECISIÓN CLASIFICACIÓN ##
# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Defino el algoritmo a utilizar
# Arboles de decisión
algoritmo = DecisionTreeClassifier(criterion='gini',
                                   min_samples_split=20,
                                   min_samples_leaf=5,
                                   max_depth=best_depth_pst,
                                   class_weight={1: 3.5})
# criterion='gini', min_samples_split=20, min_samples_leaf=5, best_depth_pst=depth, class_weight={1: 3.5}
# Entreno el modelo
algoritmo.fit(X_train, y_train)

# Realiso una predicción
y_pred = algoritmo.predict(X_test)

# Verifico la matriz de confusión
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(algoritmo.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
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
# Grafico
with open(r"tree1.dot", 'w') as f:
    f = tree.export_graphviz(algoritmo,
                             out_file=f,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('tree1.dot')
graph.write_png('tree1.png')
print("Decision Tree created successfully")
