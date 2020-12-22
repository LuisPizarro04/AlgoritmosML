from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pydot
from sklearn import tree
from sklearn.model_selection import cross_val_score
data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()
cv = KFold(n_splits=30)  # Numero deseado de "folds" que haremos
accuracies = list()
depth_range = range(1, 20)

# Parámetros
"""
n_estimators : int, default=100
    The number of trees in the forest.
    Changed in version 0.22: The default value of n_estimators changed from 10 to 100 in 0.22.

criterion : {“gini”, “entropy”}, default=”gini”
    The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and 
    “entropy” for the information gain. Note: this parameter is tree-specific.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all 
    leaves contain less than min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:
    If int, then consider min_samples_split as the minimum number.
    If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number 
    of samples for each split.
    Changed in version 0.18: Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered 
    if it leaves at least min_samples_leaf training samples in each of the left and right branches.
    This may have the effect of smoothing the model, especially in regression.
    If int, then consider min_samples_leaf as the minimum number.
    If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of 
    samples for each node.
    Changed in version 0.18: Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
    Samples have equal weight when sample_weight is not provided.

max_features : {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
    The number of features to consider when looking for the best split:
    If int, then consider max_features features at each split.
    If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
    If “auto”, then max_features=sqrt(n_features).
    If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
    If “log2”, then max_features=log2(n_features).
    If None, then max_features=n_features.
    Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    even if it requires to effectively inspect more than max_features features.

max_leaf_nodes : int, default=None
    Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. 
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    The weighted impurity decrease equation is the following:
    N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
    where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of 
    samples in the left child, and N_t_R is the number of samples in the right child.
    N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
    New in version 0.19.

min_impurity_split : float, default=None
    Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, 
    otherwise it is a leaf.
    Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19. 
    The default value of min_impurity_split has changed from 1e-7 to 0 in 0.23 and it will be removed in 0.25. 
    Use min_impurity_decrease instead.
    bootstrapbool, default=True
    Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

oob_score : bool, default=False
    Whether to use out-of-bag samples to estimate the generalization accuracy.

n_jobs : int, default=None
    The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. 
    None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 
    See Glossary for more details.

random_state : int or RandomState, default=None
    Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) 
    and the sampling of the features to consider when looking for the best split at each node 
    (if max_features < n_features). See Glossary for details.

verbose : int, default=0
    Controls the verbosity when fitting and predicting.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, 
    just fit a whole new forest. See the Glossary.

class_weight : {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
    Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed 
    to have weight one. For multi-output problems, a list of dicts can be provided in the same 
    order as the columns of y.
    Note that for multioutput (including multilabel) weights should be defined for each class of every 
    column in its own dict. 
    For example, for four-class multilabel classification weights should be 
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].

    The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies 
    in the input data as n_samples / (n_classes * np.bincount(y))

    The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap 
    sample for every tree grown.
    For multi-output, the weights of each column of y will be multiplied.
    Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is 
    specified.

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is 
    smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity 
    Pruning for details.
    New in version 0.22.

max_samples : int or float, default=None
    If bootstrap is True, the number of samples to draw from X to train each base estimator.
    If None (default), then draw X.shape[0] samples.
    If int, then draw max_samples samples.
    If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0, 1).
    New in version 0.22.
"""
for depth in depth_range:
    fold_accuracy = []
    tree_model = RandomForestClassifier(n_estimators=10,  # mientras mas mejor, pero mas lento
                                        criterion='entropy',  # entropy
                                        bootstrap=True,
                                        max_features='auto',  # Numero maximo de caract que los bosques
                                        # consideran para dividir un nodo
                                        min_samples_split=20,
                                        min_samples_leaf=5,
                                        max_depth=depth,  # profundidad del arbol(defc=auto)
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
plt.style.use("ggplot")
plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(depth_range, accuracies, label='Precisión')
plt.xlabel('Profundidad')
plt.ylabel('Precisión')
plt.title("Precisión por nivel de profundidad del Bosque")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.xticks([0, 5, 10, 15, 20])
plt.savefig('RF_grafico_prec.png', dpi=300)
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
algoritmo = RandomForestClassifier(n_estimators=20,  # mientras mas mejor, pero mas lento
                                   criterion='entropy',  # entropy
                                   bootstrap=True,
                                   max_features='auto',  # Numero maximo de caract que los bosques
                                   # consideran para dividir un nodo
                                   min_samples_split=20,
                                   min_samples_leaf=5,
                                   max_depth=best_depth_pst,  # profundidad del arbol(defc=auto)
                                   class_weight={1: 3.5})
# criterion='gini', min_samples_split=20, min_samples_leaf=5, best_depth_pst=depth, class_weight={1: 3.5}
# Entreno el modelo
algoritmo.fit(X_train, y_train)

# Realiso una predicción
y_pred = algoritmo.predict(X_test)

# Verifico la matriz de confusión
print('Precisión del Bosque Aleatorio "Clasificación" en el conjunto de entrenamiento: {:.2f}'
      .format(algoritmo.score(X_train, y_train)))
print('Precisión del Bosque Aleatorio "Clasificación" en el conjunto de prueba: {:.2f}'
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

#Guardar los datos de prueba






clf = algoritmo.estimators_
# print(algoritmo.estimators_)
print(len(algoritmo.estimators_))

fn = data.drop(['hora_ideal'], axis=1)
# Se define los datos correspondientes a la etiqueta
cn = data["hora_ideal"]

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 8), dpi=900)
for index in range(0, 5):
    tree.plot_tree(algoritmo.estimators_[index],
                   feature_names=list(data.drop(['hora_ideal'], axis=1)),
                   class_names=['a', 'b', 'c', 'd', 'e'],
                   filled=True,
                   ax=axes[index])

    axes[index].set_title('Estimator: ' + str(index), fontsize=11)
fig.savefig('RF_NN_Trees.png')


with open(r"RF_0_Tree.dot", 'w') as f:
    f = tree.export_graphviz(algoritmo.estimators_[0],
                             out_file=f,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('RF_0_Tree.dot')
graph.write_png('RF_0_Tree.png')

with open(r"RF_1_Tree.dot", 'w') as g:
    g = tree.export_graphviz(algoritmo.estimators_[1],
                             out_file=g,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('RF_1_Tree.dot')
graph.write_png('RF_1_Tree.png')

with open(r"RF_2_Tree.dot", 'w') as h:
    h = tree.export_graphviz(algoritmo.estimators_[2],
                             out_file=h,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('RF_2_Tree.dot')
graph.write_png('RF_2_Tree.png')

with open(r"RF_3_Tree.dot", 'w') as i:
    i = tree.export_graphviz(algoritmo.estimators_[3],
                             out_file=i,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('RF_3_Tree.dot')
graph.write_png('RF_3_Tree.png')

with open(r"RF_4_Tree.dot", 'w') as j:
    j = tree.export_graphviz(algoritmo.estimators_[4],
                             out_file=j,
                             max_depth=best_depth_pst,
                             impurity=True,
                             feature_names=list(data.drop(['hora_ideal'], axis=1)),
                             class_names=['a', 'b', 'c', 'd', 'e'],
                             rounded=True,
                             filled=True)

(graph,) = pydot.graph_from_dot_file('RF_4_Tree.dot')
graph.write_png('RF_4_Tree.png')
