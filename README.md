# Repo cv2_tools

Repository de partage de petits outils & script pour de la *computer vision*.


---


## inpainting.py

Script permettant pour une vidéo de produire un masque automatique sur les mains. Outil utilisé pour le billet de [blog des IA et des mains](https://post.felixjely.fr/2024/02/des-ias-et-des-mains/). L'outil se base sur Mediapipe et opencv pour detecter des mains dans l'image puis produire un masque.

Usage : `python --input path/to/vid.mp4 --size 150`

Arguments:
  - input : chemin vers la video à masqué 
  - size : (optionnelle) épaisseurs du contour du masque


Testé sur python 3.9.18 avec
```
mediapipe             0.10.9
numpy                 1.22.4
opencv-contrib-python 4.9.0.80
tqdm                  4.66.1
imageio               2.36.0
imageio-ffmpeg        0.5.1
```