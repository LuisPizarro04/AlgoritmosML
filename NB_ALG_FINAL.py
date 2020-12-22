import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"prueba_5.csv")  # archivo a procesar
print('\n\nDatos cargados correctamente!')
data.head()
dataset = pd.DataFrame(data)

# Seleccionamos las columnas
X = dataset.drop(['hora_ideal'], axis=1)
# Se deine los datos correspondientes a la etiqueta
y = dataset["hora_ideal"]

# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definimos el algortimo

algoritmo = GaussianNB()

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
# Validación cruzada
print("Validadción cruzada")
scores = cross_val_score(algoritmo, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X = data.drop(['hora_ideal'], axis=1).values
# Se deFine los datos correspondientes a la etiqueta
y = data["hora_ideal"].values
clf = algoritmo
pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X)
clf.fit(X_train2, y)
# Plotting decision region
plt.figure(figsize=(8, 5), dpi=300)
plot_decision_regions(X_train2, y, clf=clf, legend=2)
# Adding axes annotations
plt.xlabel('Características')
plt.ylabel('Objetivo')
plt.title("Naive Bayes :Límite de la región de decisión")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('NB_Limit.png', dpi=300)
plt.show()
