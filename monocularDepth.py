import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# DEPTH DETECTION
# Read Network
path_model = "models/"
#model_name = "model-f6b98070.onnx"; # MiDaS v2.1 Large
model_name = "model-small.onnx"; # MiDaS v2.1 Small

# Load the DNN model
model = cv2.dnn.readNet(path_model + model_name)

if (model.empty()):
    print("Could not load the neural net! - Check path")

# Set backend and target to CUDA to use GPU
#model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def depth_to_distance(depth):
    return -1.7 * depth + 2

# POSE DETECTION
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    imgHeight, imgWidth, channels = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    center_point = (0, 0)
    if results.pose_landmarks:
        cv2.putText(image, f'{results.pose_landmarks.landmark[4].x}, {results.pose_landmarks.landmark[4].y}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        center_point = (results.pose_landmarks.landmark[4].x * imgWidth, results.pose_landmarks.landmark[4].y * imgHeight)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # -------------- Depth map from neural net ---------------------------
    # Create Blob from Input Image
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    #blob = cv2.dnn.blobFromImage(image, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

    # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    blob = cv2.dnn.blobFromImage(image, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    depth_map = model.forward()
    
    depth_map = depth_map[0,:,:]
    depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    depth_face = depth_map[int(center_point[1]), int(center_point[0])]

    depth_face = depth_to_distance(depth_face)
    #print("Depth to face: ", depth_face)
    cv2.putText(image, "Depth in cm: " + str(round(depth_face,2)*100), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    
    # Convert the image color back so it can be displayed
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Flip the image horizontally for a selfie-view display.

    cv2.imshow('Depth map', depth_map)
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
