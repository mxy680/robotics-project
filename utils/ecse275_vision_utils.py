#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:46:49 2023

@author: zxc703
"""

import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
import tensorflow as tf

def hand_eye_transform(pos_cam, T_cam_world):
    
    '''
    Transform a position from a camera coordinate system to a world coordinate system using a given transformation matrix.
    
    Args:
        pos_cam (numpy.ndarray): A 1D NumPy array representing a position in the camera coordinate system (e.g., [x, y, z]).
        T_cam_world (numpy.ndarray): A 3x4 transformation matrix representing the transformation from camera to world coordinates.
    
    Returns:
        numpy.ndarray: A 1D NumPy array representing the transformed position in the world coordinate system.
    '''
    
    
    pos_world = np.dot(T_cam_world[:3,:3],pos_cam) + T_cam_world[:,3]
    
    return pos_world
    
def move_to(sim,desired_pos,offset=0.01,approach_height=0.1,wait_time=3):
    '''
    Move a robot to a desired position while implementing an approach and optional offset.

    Args:
        desired_pos (list): A list representing the desired position in 3D space (x, y, z) to which the robot should move.
        offset (float, optional): An offset value added to the desired z-coordinate to create a relative position for the robot.
        approach_height (float, optional): The height above the desired position to approach before reaching it.
        wait_time (float, optional): The maximum simulation time (in seconds) to wait for the robot to reach the desired position.
    
    Returns:
        None
    '''
    start_time = sim.getSimulationTime()
    approach_pos = copy.copy(desired_pos)
    approach_pos[2] = approach_pos[2]+approach_height
    desired_pos[2] = desired_pos[2]+offset
    sim.callScriptFunction('set_desired_pose',sim.getScript(1,sim.getObject('/Franka')),approach_pos)
    print("position for approach")
    while sim.getSimulationTime()-start_time < wait_time:
        pass
    start_time = sim.getSimulationTime()
    print("moving to target")
    sim.callScriptFunction('set_desired_pose',sim.getScript(1,sim.getObject('/Franka')),desired_pos)
    while sim.getSimulationTime()-start_time < wait_time:
        pass
    print("movement_completed")
    
   
def toggle_gripper(sim,wait_time=2):
    '''
    Toggle the gripper of a robot and wait for a specified duration.

    Args:
        wait_time (float, optional): The time (in seconds) to wait after toggling the gripper.
    
    Returns:
        None
    
    '''
    start_time = sim.getSimulationTime()
    print('toggling gripper')
    sim.callScriptFunction('toggle_gripper',sim.getScript(1,sim.getObject('/Franka/FrankaGripper')))
    while sim.getSimulationTime()-start_time < wait_time:
        pass

def detect_blobs(image, visualize=False):
    '''Uses OpenCV to detect blobs in an image and return a special keypoints iterable object.'''
    # Blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    # No need to filter by color since we will filter for the table
    params.filterByColor = False
    params.blobColor = 0  # 0 means black blobs
    # Filter by area
    params.filterByArea = True
    params.minArea = 20  # Minimum blob area
    # Filter by circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1  # Adjust for irregular shapes
    params.minThreshold = 0
    params.maxThreshold = 255
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(image)
    
    # Visualize blobs if requested
    if visualize:
        result_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(result_image, cmap="gray")
        plt.title("Detected Blobs")
        plt.axis("off")
        #plt.show()
    
    return keypoints

def mask_image(image):
    '''Masks out the green background to focus on the fruits.'''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])  # Green lower bound
    upper_green = np.array([85, 255, 255])  # Green upper bound
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    masked_image = cv2.bitwise_not(mask)  # Invert to keep non-green regions
    return masked_image

def blob_images(image, keypoints, roi_size=15):
    '''Returns a list of 2D pixel arrays of the regions of interest in a given image.
       Extracts square regions around the keypoints.'''
    roi = []
    centroids = []
    for kp in keypoints:
        x, y = map(int, kp.pt)
        half_size = roi_size // 2
        # Extract square ROI around the blob center
        cropped_img = image[max(0, y - half_size):min(y + half_size, image.shape[0]),
                                   max(0, x - half_size):min(x + half_size, image.shape[1])]
        # Threshold the ROI
        cropped_img[np.where(cropped_img < 55)] = 0
        roi.append(cropped_img)
        centroids.append((x, y))
    return centroids, roi


def extract_and_resize_fruits(image, centroids, roi_size=34):
    """
    Extracts and resizes individual fruits from the image to 128x128.
    Args:
        image (numpy.ndarray): Original RGB image.
        centroids (list): List of centroids [(x1, y1), (x2, y2), ...].
        roi_size (int): Size to which each fruit image should be resized.
    Returns:
        list: List of resized fruit images (128x128 pixels each).
    """
    fruits = []
    for (cx, cy) in centroids:
        # Define a square ROI around each centroid
        half_size = roi_size // 2
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, image.shape[1])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, image.shape[0])

        # Crop the ROI
        cropped_fruit = image[y_start:y_end, x_start:x_end]
        
        fruits.append(cropped_fruit)
    
    return fruits


def preprocess_fruit_images(fruits, model_input_size):
    """
    Prepares the fruit images for input into a CNN.
    Args:
        fruits (list): List of cropped fruit images (NumPy arrays).
        model_input_size (tuple): Expected input size for the model (height, width, channels).
    Returns:
        tf.Tensor: TensorFlow tensor of preprocessed images ready for the CNN.
    """
    preprocessed_images = []
    
    for fruit in fruits:
        # Resize the image to the model's expected input size
        fruit_resized = cv2.resize(fruit, (model_input_size[1], model_input_size[0]))
        
        # Normalize pixel values to [0, 1]
        fruit_normalized = fruit_resized / 255.0
        
        # Append preprocessed image
        preprocessed_images.append(fruit_normalized)
    
    # Convert list of images to a TensorFlow tensor
    images_tensor = tf.convert_to_tensor(preprocessed_images, dtype=tf.float32)
    
    return images_tensor