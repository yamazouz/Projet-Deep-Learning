# Reproduction de SqueezeNet sur CIFAR-10

Ce dépôt contient le code source permettant de reproduire l'article *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5 MB model size* que j'ai appliqué sur le dataset CIFAR-10.


## Contenu du dépôt

- **squeezenet.py** : Script principal pour l’entraînement du modèle.
- **report.md** : Rapport détaillé du projet qui répond aux questions posées.
- **requirements.txt** : Liste des dépendances nécessaires.
- **tensorboard_logs/** : Dossier généré au lancement pour stocker les logs TensorBoard.

## Instructions

1. Installer les dépendances :
    pip install -r requirements.txt

2. Lancer l’entraînement :
    python train_squeezenet.py

3. Pour visualiser les courbes d’apprentissage avec TensorBoard, lancer la commande :
    tensorboard --logdir=tensorboard_logs/

Auteur :
    Yaniss AMAZOUZ