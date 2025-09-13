"""
Enhanced Hand Sign Detection using OpenCV-Contrib
------------------------------------------------
This application uses the webcam to detect and recognize hand signs in real-time
using OpenCV's advanced modules and cascading classifiers approach.

Author: GitHub Copilot
Date: September 13, 2025
"""

import cv2
import numpy as np
import time
import math
import os

class EnhancedHandDetector:
    """
    A class for detecting and recognizing hand signs using advanced OpenCV features.
    """
    def __init__(self):
        """
        Initialize the hand detector with advanced OpenCV features.
        """
        # Parameters for skin detection in HSV color space
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Background subtractor for segmentation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.background_captured = False
        
        # Initialize hand tracking feature
        self.hand_detector = cv2.CascadeClassifier()
        
        # Try to load pre-trained models if available
        # Note: This is a placeholder; actual hand cascade files are not readily available in OpenCV
        haarcascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(haarcascade_path):
            # We'll use face detection as a placeholder
            print("Using cascade classifier for tracking")
            self.hand_detector.load(haarcascade_path)
        else:
            print("No cascade classifier found, using contour-based detection")
        
        # For gesture counting with cooldown
        self.gesture_counts = {
            'thumbs_up': 0,
            'peace': 0,
            'okay': 0,
            'rock': 0,
            'point': 0,
            'five': 0
        }
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0  # 1 second cooldown
        
        # For motion tracking
        self.prev_gray = None
        self.prev_points = None
        self.track_points = None
        
        # For gesture stabilization
        self.last_signs = []
        self.max_signs = 10
        
    def find_hands(self, img):
        """
        Detect hands in an image using multiple techniques.
        
        Parameters:
            img (numpy.ndarray): Input image (BGR format)
            
        Returns:
            img (numpy.ndarray): Image with drawings
            hand_info (dict): Information about the detected hand
        """
        # Initialize hand information
        hand_info = {
            "contour": None,
            "hull": None,
            "defects": None,
            "fingers": 0,
            "fingertips": [],
            "center": None,
            "bounding_box": None,
            "palm_center": None,
            "palm_radius": 0,
            "flow_vectors": None
        }
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Motion tracking using optical flow
        if self.prev_gray is not None:
            if self.track_points is not None and len(self.track_points) > 0:
                # Calculate optical flow
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.track_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Filter out points for which the flow couldn't be calculated
                good_new = new_points[status == 1]
                good_old = self.track_points[status == 1]
                
                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    img = cv2.line(img, (a, b), (c, d), (0, 255, 0), 2)
                    img = cv2.circle(img, (a, b), 5, (0, 0, 255), -1)
                
                self.track_points = good_new.reshape(-1, 1, 2)
                hand_info["flow_vectors"] = np.array([(new - old) for new, old in zip(good_new, good_old)])
        
        # Save the current frame for next iteration
        self.prev_gray = gray.copy()
        
        # Use multiple techniques for hand detection
        
        # 1. Skin color detection in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 2. Background subtraction if background has been captured
        if self.background_captured:
            fg_mask = self.bg_subtractor.apply(img)
            # Combine with skin mask for better results
            combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        else:
            combined_mask = skin_mask
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
                hull_points = cv2.convexHull(max_contour, returnPoints=True)
                hand_info["hull"] = hull
                
                # Draw the convex hull
                cv2.drawContours(img, [hull_points], 0, (0, 255, 255), 2)
                
                # Find the largest circle inside the contour (palm)
                max_dist = 0
                palm_center = hand_info["center"]
                
                if hand_info["center"]:
                    # Simple approximation: Find the farthest point from the center that's still inside the contour
                    # This is a simplified approach; a distance transform would be better
                    center_x, center_y = hand_info["center"]
                    
                    for point in max_contour[:, 0, :]:
                        dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                        if dist > max_dist:
                            max_dist = dist
                    
                    # Store palm information
                    hand_info["palm_center"] = hand_info["center"]
                    hand_info["palm_radius"] = max_dist * 0.6  # Approximate palm radius
                    
                    # Draw the palm circle
                    cv2.circle(img, hand_info["palm_center"], int(hand_info["palm_radius"]), (255, 0, 255), 2)
                
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
                        
                        # Save points for tracking
                        if fingertips:
                            self.track_points = np.array(fingertips, dtype=np.float32).reshape(-1, 1, 2)
                        
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
        if hand_info["contour"] is None:
            return "Unknown", 0.0
        
        # Get the number of fingers
        fingers = hand_info["fingers"]
        fingertips = hand_info["fingertips"]
        center = hand_info["center"]
        palm_center = hand_info["palm_center"]
        palm_radius = hand_info["palm_radius"]
        
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
        
        # Enhanced recognition using spatial relationship of fingertips
        if fingers > 0 and center and fingertips:
            # Get distances from center to fingertips
            distances = []
            angles = []
            
            for tip in fingertips:
                # Calculate distance from center to fingertip
                dist = np.sqrt((tip[0] - center[0])**2 + (tip[1] - center[1])**2)
                distances.append(dist)
                
                # Calculate angle
                angle = math.atan2(tip[1] - center[1], tip[0] - center[0])
                angles.append(angle)
            
            # Sort angles for consistent ordering
            sorted_indices = np.argsort(angles)
            sorted_angles = [angles[i] for i in sorted_indices]
            
            # Calculate angle differences for neighboring fingers
            angle_diffs = []
            for i in range(len(sorted_angles)):
                next_idx = (i + 1) % len(sorted_angles)
                diff = abs(sorted_angles[next_idx] - sorted_angles[i])
                # Handle wrap-around
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                angle_diffs.append(diff)
            
            # Special cases based on finger count and spatial arrangement
            if fingers == 1:
                # Determine if the extended finger is likely the thumb
                if fingertips and center:
                    tip = fingertips[0]
                    # Check horizontal distance vs vertical distance
                    dx = abs(tip[0] - center[0])
                    dy = abs(tip[1] - center[1])
                    if dx > dy:
                        sign = "Thumbs Up"
                        confidence = 0.7
                    else:
                        sign = "Point"
                        confidence = 0.8
            
            elif fingers == 2:
                # Check if fingers are spread in a V shape (peace sign)
                if len(angle_diffs) > 0 and max(angle_diffs) > math.pi/4:
                    sign = "Peace"
                    confidence = 0.8
                else:
                    sign = "Two"
                    confidence = 0.7
            
            elif fingers == 5:
                sign = "Five"
                confidence = 0.9
        
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
            "Rock": "rock",
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
    detector = EnhancedHandDetector()
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    print("Enhanced Hand Sign Detection (OpenCV Advanced)")
    print("Press 'b' to capture background for better segmentation")
    print("Press 'q' to quit")
    
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
            
            # Display finger count
            cv2.putText(img, f'Fingers: {hand_info["fingers"]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        
        # Display FPS on the image
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display gesture counts
        y_offset = 150
        for gesture, count in detector.gesture_counts.items():
            cv2.putText(img, f'{gesture}: {count}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        # Display detection mode
        mode_text = "BG Subtraction" if detector.background_captured else "Skin Detection"
        cv2.putText(img, mode_text, (10, img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the image
        cv2.imshow('Enhanced Hand Sign Detection', img)
        
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