"""
Advanced Hand Sign Detection without MediaPipe
--------------------------------------------
This application uses OpenCV to detect hand signs and gestures without requiring MediaPipe.
It's more advanced than the basic_hand_detector.py but not as accurate as the original MediaPipe-based version.

Author: GitHub Copilot
Date: September 13, 2025
"""

import cv2
import numpy as np
import time
import math

class HandSignDetector:
    """
    A class for detecting and recognizing hand signs using OpenCV.
    """
    def __init__(self):
        """
        Initialize the hand detector.
        """
        # Initialize parameters for skin detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Flag to indicate if background has been captured
        self.background_captured = False
        
        # For gesture counting and cooldown
        self.gesture_counts = {
            'thumbs_up': 0,
            'peace': 0,
            'okay': 0,
            'point': 0,
            'five': 0
        }
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0  # 1 second cooldown
        
        # Store the last few detected signs for stabilization
        self.last_signs = []
        self.max_signs = 5
        
    def find_hands(self, img):
        """
        Detect hands in an image and draw landmarks and connections.
        
        Parameters:
            img (numpy.ndarray): Input image
            
        Returns:
            img (numpy.ndarray): Image with drawings
            hand_info (dict): Information about the detected hand
        """
        # Make a copy of the image for processing
        img_copy = img.copy()
        
        # Apply background subtraction or skin detection based on mode
        if self.background_captured:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(img_copy)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Convert to HSV color space for skin detection
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
            
            # Create a mask for skin color detection
            skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize hand information
        hand_info = {
            "contour": None,
            "hull": None,
            "defects": None,
            "fingers": 0,
            "fingertips": [],
            "center": None,
            "bounding_box": None
        }
        
        # Find the largest contour which should be the hand
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough (hand, not noise)
            if cv2.contourArea(max_contour) > 1000:
                # Store the contour
                hand_info["contour"] = max_contour
                
                # Get the bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(max_contour)
                hand_info["bounding_box"] = (x, y, w, h)
                
                # Draw the bounding rectangle around the hand
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw the contour
                cv2.drawContours(img, [max_contour], 0, (0, 0, 255), 2)
                
                # Calculate the center of the contour
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    hand_info["center"] = (cX, cY)
                    cv2.circle(img, (cX, cY), 5, (255, 255, 0), -1)
                
                # Get the convex hull of the contour
                hull = cv2.convexHull(max_contour, returnPoints=False)
                hand_info["hull"] = hull
                
                # Find convexity defects
                try:
                    defects = cv2.convexityDefects(max_contour, hull)
                    hand_info["defects"] = defects
                    
                    # Initialize finger count
                    finger_count = 0
                    fingertips = []
                    
                    if defects is not None and len(defects) > 0:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i][0]
                            start = tuple(max_contour[s][0])
                            end = tuple(max_contour[e][0])
                            far = tuple(max_contour[f][0])
                            
                            # Calculate the triangle sides
                            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                            
                            # Apply cosine rule to get angle
                            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                            
                            # Ignore angles that are too large (not between fingers)
                            if angle <= np.pi / 2:  # 90 degrees
                                finger_count += 1
                                fingertips.append(end)
                                
                                # Draw the defect point
                                cv2.circle(img, far, 5, [255, 0, 0], -1)
                                
                                # Draw lines from defect point to convex points
                                cv2.line(img, start, far, [0, 255, 0], 2)
                                cv2.line(img, end, far, [0, 255, 0], 2)
                        
                        # Add 1 for the thumb (or the last endpoint)
                        finger_count = min(finger_count + 1, 5)
                        
                        # Draw fingertips
                        for tip in fingertips:
                            cv2.circle(img, tip, 8, [0, 0, 255], -1)
                        
                        hand_info["fingers"] = finger_count
                        hand_info["fingertips"] = fingertips
                        
                except:
                    pass
        
        return img, hand_info
    
    def recognize_sign(self, hand_info):
        """
        Recognize hand sign based on detected features.
        
        Parameters:
            hand_info (dict): Information about the detected hand
            
        Returns:
            sign (str): Recognized hand sign
            confidence (float): Confidence level of the recognition
        """
        # If no hand info is available, return 'Unknown'
        if not hand_info["contour"] is not None:
            return "Unknown", 0.0
        
        # Get the number of fingers
        fingers = hand_info["fingers"]
        
        # Map finger count to potential signs
        sign_map = {
            0: "Fist",
            1: "Point",
            2: "Peace",
            3: "Three",
            4: "Four",
            5: "Five"
        }
        
        # Basic sign recognition based on finger count
        sign = sign_map.get(fingers, "Unknown")
        confidence = 0.6  # Base confidence
        
        # Additional logic for specific hand signs
        if fingers == 1:
            # Check if the extended finger is the thumb
            # This is a rough approximation - without MediaPipe, it's hard to distinguish
            if hand_info["center"] and hand_info["fingertips"]:
                if len(hand_info["fingertips"]) > 0:
                    fingertip = hand_info["fingertips"][0]
                    center = hand_info["center"]
                    
                    # Rough check if the fingertip is to the side of the center
                    if abs(fingertip[0] - center[0]) > abs(fingertip[1] - center[1]):
                        sign = "Thumbs Up"
                        confidence = 0.7
        
        # Add the sign to the history for stabilization
        self.last_signs.append(sign)
        if len(self.last_signs) > self.max_signs:
            self.last_signs.pop(0)
        
        # Stabilize the detection by taking the most common sign in the history
        from collections import Counter
        if self.last_signs:
            sign_counter = Counter(self.last_signs)
            stable_sign = sign_counter.most_common(1)[0][0]
            stable_confidence = sign_counter.most_common(1)[0][1] / len(self.last_signs)
            
            # Only change the sign if we're confident in the new detection
            if stable_confidence > 0.6:
                sign = stable_sign
                confidence = min(stable_confidence + 0.2, 0.95)  # Boost confidence but cap at 0.95
        
        return sign, confidence
    
    def count_gesture(self, sign):
        """
        Count specific gestures when they are detected with cooldown.
        
        Parameters:
            sign (str): Detected sign
            
        Returns:
            None
        """
        # Map sign to gesture counter
        gesture_map = {
            "Thumbs Up": "thumbs_up",
            "Peace": "peace",
            "Okay": "okay",
            "Point": "point",
            "Five": "five"
        }
        
        # Check if the sign is one we want to count and if the cooldown has passed
        current_time = time.time()
        if sign in gesture_map and current_time - self.last_gesture_time >= self.gesture_cooldown:
            self.gesture_counts[gesture_map[sign]] += 1
            self.last_gesture_time = current_time
    
    def capture_background(self):
        """Toggle background capture mode"""
        self.background_captured = True
        return self.background_captured

def main():
    """
    Main function to run the hand sign detection application.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize detector
    detector = HandSignDetector()
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    print("Advanced Hand Sign Detection (OpenCV-only version)")
    print("Press 'b' to capture background for better segmentation.")
    print("Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break
            
        # Mirror the image horizontally for a more natural interaction
        img = cv2.flip(img, 1)
        
        # Find hands in the frame
        img, hand_info = detector.find_hands(img)
        
        # Recognize hand sign if hand was detected
        if hand_info["contour"] is not None:
            sign, confidence = detector.recognize_sign(hand_info)
            
            # Count gestures
            detector.count_gesture(sign)
            
            # Display detected sign and confidence
            cv2.putText(img, f'{sign} ({confidence:.2f})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        
        # Display FPS on the image
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display gesture counts
        y_offset = 110
        for gesture, count in detector.gesture_counts.items():
            cv2.putText(img, f'{gesture}: {count}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        # Display detection mode
        mode_text = "BG Subtraction" if detector.background_captured else "Skin Detection"
        cv2.putText(img, mode_text, (10, img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the image
        cv2.imshow('Advanced Hand Sign Detection', img)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit if 'q' is pressed
            break
        elif key == ord('b'):
            # Capture background if 'b' is pressed
            detector.capture_background()
            print("Background captured. Using background subtraction method.")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()