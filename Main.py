#programme principal résolution ACP

from math import*
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt

##X = np.array([[16, 16],
##              [14, 7],
##              [6, 15],
##              [8, 10]])

numRows = int(input("Combien de lignes?") )
numColumns = int(input("Combien de colonnes?"))

X = []
for i in range(numRows):
    X.append([0] * numColumns)
    for j in range(numColumns):
        X[i][j] = float(input("Entrer les valeurs de la matrice"))
        print(X)

print("La matrice de départ X vaut: \n", X, "\n")

X_std = StandardScaler().fit_transform(X)
print("Remise à l'échelle des vecteurs caractéristiques pour qu'ils aient tous la même échelle: \n", X_std, "\n")

features = X_std.T
correlation_matrix = np.cov(features)
print("La matrice de corrélation (après mise à l'echelle) vaut: \n", correlation_matrix, "\n")


eigVects_after_scaled, eigVals_after_scaled = eig(correlation_matrix)
print("Les valeurs propres de la matrice de corrélation (après mise à l'echelle) sont: \n", eigVects_after_scaled, "\n")
print("Les vecteurs propres de la matrice de corrélation (après mise à l'echelle) sont: \n", eigVals_after_scaled, "\n")

pca = PCA(n_components = 2)
principal_components = pca.fit_transform(X_std)
new_X = pd.DataFrame(data = principal_components, columns = ['Composante principale 1', 'Principale composante 2'])
new_X.head()
print("La matrice 'Psi' sous forme de tableau vaut: \n", new_X)

##pourcentage_variance = np.round(pca.explained_variance_ratio_*100, decimals = 1)
##label = ['PC' + str(x) for x in range(1, len(pourcentage_variance) + 1)]
##plt.bar(x = range(1, len(pourcentage_variance) + 1), height = pourcentage_variance, tick_label = label)
##plt.xlabel("Composantes principales")
##plt.ylabel("Pourcentage de variance expliqué")
##plt.title("Graphe de pourcentage des composantes principales")
###plt.show()
##

color = 'red'
plt.figure(figsize = (2, 2))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c = color, cmap = 'viridis', alpha = 1)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Plan principal 'psi1' et 'psi2'")
plt.show()





