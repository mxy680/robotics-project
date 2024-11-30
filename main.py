import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import utils.ecse275_vision_utils as util
import utils.my_functions_solution as func
import json

plt.close("all")

f = 0.020  # focal length in meters
pixels_per_inch = 560.0165995731867
z = 0.805  # vertical distance to the centerpoint of the blocks on the table

vision_mode = "RGB"  # or "RGB"

client = zmq.RemoteAPIClient()
sim = client.getObject('sim')

drop_target = sim.getObject('/drop_target')
droppt = sim.getObjectPose(drop_target, -1)

camera_handle = sim.getObject("/Vision_sensor")

# Capture image from the vision sensor
if vision_mode == "gray":
    image, resolution = sim.getVisionSensorImg(camera_handle, 1)
    image = np.array(bytearray(image), dtype='uint8').reshape(resolution[0], resolution[1])
elif vision_mode == "RGB":
    image, resolution = sim.getVisionSensorImg(camera_handle, 0)
    image = np.array(bytearray(image), dtype='uint8').reshape(resolution[0], resolution[1], 3)
else:
    print("Invalid vision mode!")

image = np.flip(image, axis=1)

if vision_mode == "gray":
    plt.imshow(image, cmap="binary")
elif vision_mode == "RGB":
    plt.imshow(image)
    #plt.show()

# Detect blobs
masked_image = util.mask_image(image)
keypoints = util.detect_blobs(masked_image, visualize=True)

# Get centroids and ROI
centroids, roi = util.blob_images(masked_image, keypoints)

# Extract and resize fruits
resized_fruits = util.extract_and_resize_fruits(image, centroids)

# Load TensorFlow model
model_path = "models/model.keras"  # Replace with your actual TensorFlow model file path
model = tf.keras.models.load_model(model_path)

# Get model's input shape (exclude batch size)
model_input_shape = model.input_shape[1:]  # e.g., (128, 128, 3)

# Preprocess the fruits
preprocessed_fruits = util.preprocess_fruit_images(resized_fruits, model_input_shape)

# Predict using the CNN
predictions = model.predict(preprocessed_fruits)

# Load class indices from json
class_indices = json.load(open("data/class_indices.json", "r"))
for i, pred in enumerate(predictions):
    predicted_label = [key for key,value in class_indices.items() if value == np.argmax([pred], axis=1)[0]]
    
    plt.figure()
    plt.imshow(resized_fruits[i])  # Show the original image
    plt.title(f"Predicted Class: {predicted_label}")
    plt.show()

exit()

# Sort detected objects by predicted order
order_sequence = np.argsort(predictions)  # Sort by ascending order

T_cam_world = np.array(sim.getObjectMatrix(camera_handle, -1)).reshape(3, 4)
pos_cam_list = []
pos_world_list = []

# Compute positions in camera and world coordinates
for i in order_sequence:
    pos_cam_list.append(func.compute_pos_from_pix(centroids[i], resolution[0], f, pixels_per_inch, z))

for pos_cam in pos_cam_list:
    pos_world_list.append(util.hand_eye_transform(pos_cam, T_cam_world))

# Execute robot actions to pick and place objects
for i in range(len(pos_world_list)):
    print("Picking...", predictions[order_sequence[i]])
    util.move_to(sim, list(pos_world_list[i]), offset=0.02)
    util.toggle_gripper(sim)
    
    droppt[2] = droppt[2] + 0.015
    util.move_to(sim, droppt, offset=0.02)
    util.toggle_gripper(sim)
    
    reset_point = [droppt[0], droppt[1], droppt[2] + 0.1]
    util.move_to(sim, reset_point, approach_height=0)
