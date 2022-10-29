#programme principal résolution ACP

from math import*
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt

##tableau test
##X = np.array([[16, 16],
##              [14, 7],
##              [6, 15],
##              [8, 10]])

##X = np.array([[-4, -5],
##              [0, 11],
##              [3, 10],
##              [6, 12],
##              [8, -4],
##              [5, 0]])

##X = np.array([[167, 1, 163,  23, 41, 8, 6, 6],
## [162, 2, 141, 12, 40, 12, 4, 15],
## [119, 6, 69, 56, 39, 5, 13, 41],
## [ 87, 11, 63, 111, 27, 3, 18, 39],
## [103, 5, 68, 77, 32, 4, 11, 30],
## [111, 4, 72, 66, 34, 6, 10, 28],
## [130, 3, 76, 52, 43, 7, 7, 16],
## [138, 7, 117, 74, 53, 8, 12, 20]])

numRows = int(input("Combien de lignes?") ) #on demande à l'utilisateur de rentrer le nombre de ligne de la matrice
numColumns = int(input("Combien de colonnes?")) #on demande à l'utilisateur de rentrer le nombre de colonne de la matrice

#tableau dynamique
X = []
for i in range(numRows):
    X.append([0] * numColumns)
    for j in range(numColumns):
        X[i][j] = float(input("Entrer les valeurs de la matrice"))
        print(X)
X = np.array(X)
print("La matrice de départ X vaut: \n", X, "\n")

X_std = StandardScaler().fit_transform(X)
print("Données centrées réduites. En d'autres termes, la matrice 'Z' vaut: \n", X_std, "\n")

features = X_std.T

corr_matrix_without_scale = np.corrcoef(X.T)
vals, vects = eig(corr_matrix_without_scale)
print("Les valeurs propres de la matrice de corrélation sont: \n", vals, "\n")

correlation_matrix = np.cov(features)
print("La matrice de corrélation (après mise à l'echelle) vaut: \n", correlation_matrix, "\n")

eigVals_after_scaled, eigVects_after_scaled = eig(correlation_matrix)
print("Les valeurs propres de la matrice de corrélation (après mise à l'echelle) sont: \n", eigVals_after_scaled, "\n")
print("Les vecteurs propres de la matrice de corrélation (après mise à l'echelle) sont: \n", eigVects_after_scaled, "\n")

psi = np.dot(X_std, eigVects_after_scaled)  #calcul de la matrice 'psi'

n_components = 2
pca = PCA(n_components = n_components)

principal_components = pca.fit_transform(X_std)
new_X = pd.DataFrame(data = principal_components, columns = ['Composante principale 1', 'Principale composante 2'])
new_X.head()    #matrice 'Psi' sous forme de tableau grâce à Pandas
print("La matrice 'Psi' sous forme de tableau vaut: \n", new_X)

##print(vals)
##print(eigVects_after_scaled[0])
##print(eigVects_after_scaled[1])

#calcul des valeurs du cercle de corrélation
liste1 = []
liste2 = []

for i in eigVects_after_scaled[:, 0]:
    coord1 = sqrt(vals[1]) * i
    liste1.append(coord1)

for j in eigVects_after_scaled[:, 1]:
    coord2 = sqrt(vals[0]) * j
    liste2.append(coord2)

liste1 = np.array(liste1)
liste2 = np.array(liste2)
liste1 = np.vstack([liste1, liste2])
#print(liste1)

#Graphe de pourcentage des composantes principales
pourcentage_variance = np.round(pca.explained_variance_ratio_*100, decimals = 1)
label = ['PC' + str(x) for x in range(1, len(pourcentage_variance) + 1)]
plt.bar(x = range(1, len(pourcentage_variance) + 1), height = pourcentage_variance, tick_label = label)
plt.xlabel("Composantes principales")
plt.ylabel("Pourcentage de variance expliqué")
plt.title("Graphe de pourcentage des composantes principales")
#plt.show()


#plan principal
color = 'red'
#n = ["ind1", "ind2", "ind3", "ind4"]   #tableau rentré de manière statique
n = []  #tableau rentré de manière dynamique
for i in range(numRows):
    n.append("Ind" + str(i + 1))

plt.figure(figsize = (2, 2))    #dimensionnement de la fenêtre
plt.scatter(principal_components[:, 0], principal_components[:, 1], c = color, cmap = 'viridis', alpha = 1)
for i, txt in enumerate(n):
    plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1])) #place les annotation "Ind1, Ind2, ..., Indn" sur le graphe des plans principaux

plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.grid(linestyle='--')
plt.title("Plan principal 'psi1' et 'psi2'")
#plt.show()  #on affiche les graphes

#cercle de corrélation
theta = np.linspace(0, 2*np.pi, 100)

r = np.sqrt(1.0)

x1 = r*np.cos(theta)
x2 = r*np.sin(theta)


fig, ax = plt.subplots(1)
ax.plot(x1, x2)
plt.scatter(liste1[0, :], liste1[1, :], c = color, cmap = 'viridis', alpha = 1)

print(eigVects_after_scaled.shape[0])

liste_temp = []
for i in range(eigVects_after_scaled.shape[0]):
    liste_temp.append("Ind" + str(i + 1))

for i, txt in enumerate(liste_temp):
    plt.annotate(txt, (liste1[0, i], liste1[1, i]))
ax.set_aspect(1)

plt.xlim(-1.25, 1.25)
plt.ylim(-1.25, 1.25)

plt.grid(linestyle = '--')
plt.title("Cercle de corrélation", fontsize = 8)
plt.show()

