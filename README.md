<h1>Système de détection des intrusions pour le trafic réseau</h1>
L'utilisation des modèles de classification binaire pour distinguer le trafic malveillant du trafic bénin au niveau du réseau offre des avantages importants pour les entreprises en matière de cybersécurité.<br>
L'objectif de ce projet est de développer un système de classification binaire capable de détecter le trafic réseau malveillant ou bénin en analysant les données du réseau telles que le protocole de communication, la durée, les ports de source et destination, les nombres d'octets échangées, etc.
<h2>Etape 1: extraction, chargement et transformation des données sur l'infrastructure cloud de AWS </h2>
Tout d'abord, on commence par l'extraction des données depuis plusieurs sources: logs, datasets:
<ul>
  <li>CICIDS 2018 </li>
  <li>KDD CUP 1999</li>
  <li>UNSW-NB 15</li>
  <li>D'autres datasets des données du traffic réseau + logs</li>
</ul>
<div>

Les données sont stockées au niveau du système de gestion des fichiers S3 au niveau de l'infrastructure cloud de AWS.<br>
Ensuite, on procède au traitement des données au niveau du datawarehouse Redshift afin de centraliser les données et créer le dataset final qui va servir à l'apprentissage.
Au final, on obtient un jeu de données balancé avec deux types de trafic: (0: bénin) et (1: malveillant), ceci va permettre au modèle de bien apprendre à classifier les deux classes et de réduire le biais et le sur-apprentissage qui pourrait conduire à de fausses prédictions. <br>
</div>

<img src="images/balanced_dataset.PNG" width="512"/>

<h2>Etape 2: Apprentissage des modèles de classification binaire et évaluation</h2>
Dans cette étape, on utilise le langage python pour entraîner les modèles de classification et ce en utilisant les packages: scikit-learn, pandas, numpy.
<h3>2-1-Préparation du jeu de données</h3>
<ol>
  <li>Suppression des doublons</li>
  <li>Traitement des valeurs nulles et NAN</li>
  <li>Choix des features importants pour l'apprentissage</li>
  <li>Etude des corrélations entre les variables</li>
  <li>Etude de la variance des variables</li>
  <li>Traitement des variables infinies</li>
  <li>Création des échantillons équilibrés des classes binaires: bénigne, maligne</li>
  <li>Encodage des variables catégorielles</li>
  <li>Normalisation du dataset</li>
</ol>
<h3>2-2-Entraînement des modèles de classification</h3>
<ol>
<li>Préparation des jeux de données: training, testing, validation.</li>
<li>Choix des hyperparamètres en utilisant la méthode du grid search.</li>
<li>Entraînement des modèles de machine learning. </li>
  
<ol>
  <li>Random forest</li>
  <li>Logistic regression</li>
  <li>XGboost</li>
  <li>SVM</li>
  <li>Gradient boosting classifier</li>
  <li>Réseau de neuronnes composé de deux couches</li>
</ol>
</ol>
<h3>2-3-Evaluation des modèles de classification en utilisant les métriques: </h3>
<ol>
  <li>Accuracy (précision globale)</li>
  <li>Precision (taux de vrais positifs parmi les prédictions positives)</li>
  <li>Recall (taux de détection des positifs réels)</li>
  <li>F1-Score (harmonisation entre précision et rappel)</li>
  <li>Courbe ROC (évaluer le compromis entre le rappel et le taux de faux positifs)</li>
</ol>
<h3>2-4-Choix du meilleur modèle de prédiction</h3>
<div>
Nous allons comparé le précision de chaque modèle de classification afin de choisir le meilleur classifier.<br>
  
</div>
<br>
<img src="images/compare_models_acc.PNG" width="900"/>

On remarque que le modèle Xgboost offre la meilleure prédiction sur le jeu de données.<br>


<h3>2-5-Mesure de la performance du modèle choisi et interprétation des résultats</h3>

Dans ce qui suit, nous allons évaluer les performances du modèle choisi sur le jeu de données de validation sous forme de modélisations graphiques.<br>
Nous commençons par le calcul de la matrice de confusion qui va nous permettre de mesurer l'exactitude, précision et confiance du modèle à bien classifier le traffic et prédire la classe correspondante au traffic réseau. <br>
Pour clarifier les idées au niveau de ces mesures, nous introduisons certains termes ici :
<ul>
  <li>True Positive (TP) : ce sont les observations positives qui ont été prédites correctement par le modèle (ou pour simplifier, les observations prédites comme étant “oui” et étant véritablement “oui”).</li>
  <li>True Negative (TN) : de manière similaire, ce sont les observations négatives correctement prédites par le modèle (les observations prédites comme “non” et étant réellement “non”).</li>
<li>False Positive (FP) : ce sont les observations négatives prédites (de manière erronée) par le modèle comme étant positives (un vrai “non” prédit comme un “oui”).</li>
  <li>False Negative (TN) : ce sont les observations positives prédites comme étant négatives par le modèle (un vrai “oui” prédit comme un “non”).</li>
</ul>
<h4>2-5-1-La mesure de l'accuracy</h4>
L’accuracy, qui est la mesure de performance d’un modèle la plus intuitive, peut être définie à partir de ces termes : il s’agit tout simplement du ratio des observations correctes prédites sur le total des observations.<br>
Soit : accuracy = (TP+TN)/(TP+TN+FP+FN). <br>
C’est une métrique très efficace dans le cas de dataset équilibré.<br>
Pour notre projet: l'accuracy est égale à 0.97 

<h4>2-5-2-La matrice de confusion</h4>

Les matrices de confusions sont des tables qui permettent de visualiser la performance d’un modèle en affichant les mesures des TP, TN, FP, et FN. Toutes les observations qui se situent sur la diagonale de la matrice ont été correctement prédites par le modèle, tandis que les observations qui ne se situent pas sur la diagonale correspondent à des erreurs du modèle. Un modèle parfait aurait donc l’ensemble de ses prédictions sur la diagonale d’une matrice de confusion.<br>
La matrice de confusion relative à notre modèle est présentée comme suit:

<img src="images/confusion_matrix.PNG" width="512"/>

<h4>2-5-3-Modélisation de la distribution des prédictions selon les classes binaires</h4>


Le diagramme en barres représente la répartition des classes prédites par le modèle sur le jeu de données de validation, on remarque que les classes sont presque égales et que le modèle réussit à bien classifier le traffic normal et le distinguer du traffic malveillant.<br>

<img src="images/predictions_barplot.PNG" width="512"/><br>
De même, le diagramme en camembert représente plus précisément le pourcentage de prédiction de chaque classe. Vu que l'apprentissage s'est effectué sur un dataset équilibré, les pourcentages sont équilibrés également.<br>

<img src="images/predictions_pieplot.PNG" width="512"/>

<h4>2-5-4-Mesure de la confiance globale du modèle</h4>
Le modèle attribue une probabilité à chaque classe (par exemple, 0,8 = 80 % de confiance signifie que l'échantillon est malveillant). Un seuil (généralement 0,5) est utilisé pour classer les échantillons :
<ul>
  <li>Classe prédite = 1 si probabilité ≥0,5</li>
  <li>Classe prédite = 0 si probabilité <0,5</li>
</ul>
    
Ce graphique reflète les résultats de ce seuil de décision.
La classe malveillante a des probabilités systématiquement plus élevées, le modèle est capable de distinguer les deux classes en toute confiance.

<img src="images/distribution_proba.PNG" width="512"/>

Plus globalement, la distribution des probabilités de prédiction du modèle pour chaque classe est représentée dans le diagramme circulaire suivant: <br>

<img src="images/confiance_globale.PNG" width="512"/>

<h4>2-5-5-Temps d'exécution du modèle en fonction du nombre des échantillons</h4>
<div>Afin de garantir une meilleure performance du modèle, on va calculer son temps d'exécution en fonction du nombre des échantillons. Ceci, dans l'objectif de pouvoir déployer le modèle dans un système de détection d'intrusion en temps réel et dans les systèmes de monitoring du traffic réseau.<br>
Le modèle présente un temps d'exécution très réduit, permettant ainsi une utilisation efficace pour des détections en temps réel sur des flux de données de streaming ou en mode batch.<br>
</div>
<br>
<img src="images/temps_execution.PNG" width="512"/>
<h3>Références</h3>

<ul>
  <li>http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html</li>
  <li>https://www.unb.ca/cic/datasets/ids-2017.html</li>
  <li>https://research.unsw.edu.au/projects/unsw-nb15-dataset</li>

</ul>
