import cv2  # importation de OpenCV
import sys  # pour sys.exit()
import matplotlib.pyplot as plt  # pour afficher l'image

# Étape 1 – Chargement et préparation de l’image
img_bgr = cv2.imread(r"C:\Users\paola\Downloads\Images de test pour la decetion de visages\ff.jpg")  # chargement de l'image
if img_bgr is None:  # vérification du chargement
    print("Erreur: impossible de charger l'image.")
    sys.exit()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # conversion BGR->RGB pour affichage
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # conversion en niveaux de gris

plt.imshow(img_rgb)  # affichage de l'image originale
plt.title("Image originale")
plt.axis("off")
plt.show()

# Étape 2 – Détection des visages
face_cascade = cv2.CascadeClassifier(r"C:\Users\paola\Downloads\classifieurs\haarcascade_frontalface_alt2.xml")  # classifieur frontal
profile_cascade = cv2.CascadeClassifier(r"C:\Users\paola\Downloads\classifieurs\haarcascade_profileface.xml")  # classifieur profil

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # détection visage frontal
profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # détection profil (orientation entraînée)

flipped = cv2.flip(gray, 1)  # retournement horizontal
profiles_flipped = profile_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5)  # détection profil opposé

# Remise des profils retournés dans l'image originale
list_face = list(faces) + list(profiles)
W = gray.shape[1]
for (x, y, w, h) in profiles_flipped:
    x_corrected = W - x - w
    list_face.append((x_corrected, y, w, h))

# Étape 3 – Affichage des résultats
img_draw = img_rgb.copy()  # on dessine sur l'image couleur destinée à Matplotlib
for (x, y, w, h) in list_face:
    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)  

print("Nombre total de visages détectés :", len(list_face))  # affichage du nombre total

plt.imshow(img_draw)  # affichage final
plt.title("Visages détectés")
plt.axis("off")
plt.show()

# Étape 4 – Amélioration de la visualisation
resized = cv2.resize(img_draw, (0, 0), fx=0.25, fy=0.25)  # réduction taille 25%
plt.imshow(resized)
plt.title("Image réduite (zoom)")
plt.axis("off")
plt.show()

# Essayez d’augmenter l’épaisseur du rectangle (thickness=4 ou 6) et observez la différence.
cv2.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), thickness=6)  # Épaisseur 4
plt.imshow(resized)
plt.title("Image réduite (changement de thickness)")
plt.axis("off")
plt.show()


## Bonus – Améliorations et extensions

# Numérotation des visages détectés
# Pour chaque visage détecté, un numéro a été affiché au-dessus du rectangle rouge à l’aide de `cv2.putText()` :

for i, (x, y, w, h) in enumerate(list_face):
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img_rgb, f"{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
plt.imshow(img_rgb)
plt.title("Visages détectés avec numéro")
plt.axis("off")

## Detection en temps réel avec webcam

# Initialisation de la capture vidéo depuis la webcam (index 0)
cap = cv2.VideoCapture(0)

# Boucle principale pour la détection en temps réel
while True:
    ret, frame = cap.read()  # Capture une image depuis la webcam
    if not ret:
        break  # Si la capture échoue, on quitte la boucle

    # Conversion de l'image en niveaux de gris (requis pour Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Parcours des visages détectés
    for i, (x, y, w, h) in enumerate(faces):
        # Dessin d'un rectangle rouge autour du visage
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Affichage du numéro du visage au-dessus du rectangle
        cv2.putText(frame, f"{i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Affichage de l'image annotée dans une fenêtre
    cv2.imshow("Détection en temps réel", frame)

    # Sortie de la boucle si l'utilisateur appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()

#Extension : Détection avec DNN (Deep Neural Network)
#Une version améliorée peut être implémentée avec le module cv2.dnn et un modèle pré-entraîné (ex. res10_300x300_ssd_iter_140000.caffemodel). 
#Cette méthode est plus robuste et moderne que les classifieurs Haar.


# Chargement du modèle DNN pré-entraîné (ResNet-10 SSD)
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\paola\Downloads\deploy.prototxt.txt",  # Fichier de configuration du réseau telechargé depuis git
    r"C:\Users\paola\Downloads\res10_300x300_ssd_iter_140000.caffemodel"  # Poids du modèle
)

# Initialisation de la capture vidéo depuis la webcam (index 0 = webcam par défaut)
cap = cv2.VideoCapture(0)

# Boucle de traitement en temps réel
while True:
    ret, frame = cap.read()  # Capture une image (frame) depuis la webcam
    if not ret:
        break  # Si la capture échoue, on quitte la boucle

    # Récupération des dimensions de l'image
    h, w = frame.shape[:2]

    # Prétraitement de l'image pour le modèle DNN
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )  # Normalisation et redimensionnement

    net.setInput(blob)  # Envoi du blob au réseau
    detections = net.forward()  # Exécution de la détection

    # Parcours de toutes les détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Score de confiance de la détection
        if confidence > 0.5:  # Seuil de confiance minimal
            # Extraction des coordonnées de la boîte englobante
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Dessin du rectangle autour du visage détecté
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Affichage du score de confiance
            label = f"{confidence * 100:.1f}%"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Affichage de l'image annotée
    cv2.imshow("Détection DNN", frame)

    # Sortie de la boucle si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
