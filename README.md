# D-tection-d-objets-image
Segmentation en composantes connexes
1) Lecture + prétraitement de l’image

Il ouvre l’image donnée en argument (sys.argv[1]).

Il en fait une copie en niveaux de gris (convert('L')).

Il garde aussi une version numpy de l’image grise originale (image2b) (utile pour la FFT mais c’est commenté).

Ensuite il fait un pipeline :

Flou fort (BoxBlur(20))

Réduction de taille (division par fct=3) → plus rapide et plus robuste au bruit

Binarisation “locale” par blocs via adjust_contrast() :

découpe l’image en blocs 150×150

calcule la moyenne du bloc comme seuil

met les pixels en blanc (255) si < moyenne - 15, sinon noir (0)

puis dilate/élargit les zones avec deux MaxFilter (effet “grossissement”/fermeture)

Ensuite :

Il détecte les bords sur l’image binaire (FIND_EDGES)

Il “superpose” ces bords sur l’image couleur originale et sauvegarde test.png :

np.maximum(edges, image) → les contours ressortent

2) Segmentation en composantes connexes (hk_cluster)

hk_cluster(np.array(image2), 100) fait l’équivalent d’un connected-component labeling (étiquetage de composantes) sur l’image binaire :

Il crée une matrice R où les pixels “au-dessus du threshold” deviennent -1 sinon 0.

Puis il parcourt l’image et remplace les -1 par des labels 1,2,3… (en regardant seulement le voisin du haut et de gauche).

Il fusionne des labels quand deux zones se rejoignent (méthode un peu “maison”, pas union-find complet).

Ensuite, pour chaque composante trouvée (chaque “particule”) :

Il ignore les petites (len(x) < 35)

Calcule le centre de masse (com)

Estime des diamètres en regardant les pixels de bord :

pour chaque pixel du bord, il calcule tmp = 2 * distance(pixel, centre)

il collecte une liste d puis :

diam = moyenne(d) (diamètre moyen)

diam_std = std(d) (dispersion)

d_min et d_max (min/max observés)

Calcule une orientation via la covariance des coordonnées → PCA :

eigenvectors = axes principaux (grand axe / petit axe)

Stocke aussi l’aire = nombre de pixels de la composante

La fonction renvoie : centres, diamètres, écart-types, min/max, orientations, aires.

3) Remise à l’échelle + annotation sur l’image finale

Pour chaque particule détectée :

Il remet les mesures à l’échelle de l’image originale (diam *= fct, etc.).

Il convertit en nanomètres via scale = 50/1150 (donc 1150 pixels ≈ 50 nm).

Il définit :

d_min = diam - diam_std

d_max = diam + diam_std

Il dessine :

une ligne rouge selon un axe (avec longueur d_min)

une autre ligne rouge selon l’autre axe (avec longueur d_max)

un cercle bleu au centre

du texte: index + D (nm): <diamètre> +/- <std>

Résultat final : result.jpg.

Détails importants / bizarreries

Dans hk_cluster, il y a un commentaire “x et y semblent inversés” : c’est un point sensible (numpy renvoie (row=y, col=x)), et le code mélange parfois x/y et i/j.

La segmentation est basée sur un threshold sur une image déjà binarisée, donc le threshold=100 n’est pas vraiment un seuil “physique” : c’est juste pour distinguer 0 et 255.

La partie FFT avec OpenCV est commentée.

Si tu me dis quel type d’images tu analyses (microscopie ? particules ?), je peux aussi te dire si les choix (blur 20, blocs 150, seuil moyenne-15, filtre max) sont adaptés et comment améliorer la détection/mesure.
