import cv2
import argparse
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions # Utilisé pour drawing de bones (l 53)
from mediapipe.framework.formats import landmark_pb2

import numpy as np

from tqdm import tqdm
import imageio


def draw_landmarks_on_image(rgb_image, detection_result, size):
  """
  draw_landmarks_on_image: fonction qui permet de produire l'image du masque en fonction de ce que l'algo de mediapipe a calculé
  
  :params rgb_image: image d'input (@type Image cv2)
  :params : donnée renvoyées par Mediapipe (@type vision.HandLandmarker)

  :return: Image du masque (@type Image cv2)

  """
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  mask_inpainting = np.zeros(annotated_image.shape)



  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]


    # Si jamais on veut rendre les bones de la mains une fonction existe déjà dans mediapipe
    # solutions.drawing_utils.draw_landmarks(
    #   mask_inpainting,
    #   hand_landmarks_proto,
    #   solutions.hands.HAND_CONNECTIONS,
    #   solutions.drawing_styles.get_default_hand_landmarks_style(),
    #   solutions.drawing_styles.get_default_hand_connections_style())



    # Notre masque custom

    """
    reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=fr
    Points séléctionnés : 
    0 = poignet
    4 = bout du pouce
    12 = bout du majeur
    20 = bout de l'oriculaire 
    """

    pts = []
    for index, value in enumerate([0, 4, 12, 20]): # Points selectionés
        x = int(hand_landmarks[value].x * width)
        y = int(hand_landmarks[value].y * height)
        pts.append([x,y])
        cv2.circle(mask_inpainting, (x,y), 10, (255, 255, 255), -1 )

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))

    # Drawing du masque fillPoly rempli la zone, polyline dessine les contours
    cv2.fillPoly(mask_inpainting, [pts], (255,255,255))
    cv2.polylines(mask_inpainting, [pts], True, (255,255,255), size)    

  return mask_inpainting



def main():
  # Parser
  parser = argparse.ArgumentParser(description='Optional app description')

  parser.add_argument('--input',
                    help='Input video file for drawing mask')
  parser.add_argument('--size', default=30, type=int, nargs='?',
                    help='Size for mask outline')
  

  args = parser.parse_args()

  # Recuperation de l'arg parser
  input_path = Path(args.input)
  size_arg = int(args.size)


  # Initalize des readers 
  # On utilise imageio pour l'import du fichier mp4 et sont export en h264 (pas possible avec le writer opencv standard)
  reader = imageio.get_reader(input_path)
  fps = reader.get_meta_data()['fps']


  # Landmark model chargement (mediapipe sdk)
  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)

  # On construit le writer (enregistrment du fichier)
  writer_path = Path(input_path.parent.absolute(), input_path.stem + "_outputs.mp4")
  writer = imageio.get_writer(str(writer_path.resolve()), fps=fps)

  with tqdm(total=reader.count_frames()) as pbar:
      for im in reader:
          pbar.update(1)
          #image = mp.Image.create_from_file(str(img.resolve()))
          image =  mp.Image(image_format=mp.ImageFormat.SRGB, data=im.astype(np.uint8))
          # STEP 4: Detect hand landmarks from the input image.
          detection_result = detector.detect(image)

          # STEP 5: Process the classification result. In this case, visualize it.
          annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result, size_arg)

          
          writer.append_data(annotated_image.astype(np.uint8))

      writer.close()

      print(f"Enregistrment du mask à : {writer_path}")




# Startup
if __name__ == "__main__":
    main()
