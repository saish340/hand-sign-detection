"""
Real-time Hand Sign Detection using OpenCV and MediaPipe
--------------------------------------------------------
This application uses the webcam to detect and recognize hand signs in real-time.
It leverages MediaPipe for hand tracking and custom logic for sign recognition.

Author: GitHub Copilot
Date: September 13, 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

class HandSignDetector:
    """
    A class for detecting and recognizing hand signs using MediaPipe.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe Hands.
        
        Parameters:
            static_image_mode (bool): If True, treats input images as a batch of static images.
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence for hand detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence for hand tracking to be considered successful.
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                        max_num_hands=max_num_hands,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Gesture counters
        self.gesture_counts = {
            'thumbs_up': 0,
            'peace': 0,
            'okay': 0,
            'rock': 0,  # Rock & Roll sign
            'point': 0
        }
        
        # For gesture cooldown to avoid multiple counts
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0  # 1 second cooldown

    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and optionally draw landmarks and connections.
        
        Parameters:
            img (numpy.ndarray): Input image (BGR format)
            draw (bool): Whether to draw landmarks and connections on the image
            
        Returns:
            img (numpy.ndarray): Image with or without drawings
            results: MediaPipe hand detection results
        """
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if hands were detected and drawing is enabled
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        
        return img, results

    def find_positions(self, img, hand_index=0, results=None):
        """
        Find hand landmarks positions.
        
        Parameters:
            img (numpy.ndarray): Input image
            hand_index (int): Index of the hand (0 for first detected hand)
            results: MediaPipe hand detection results
            
        Returns:
            landmark_list (list): List of landmark positions [id, x, y]
        """
        h, w, c = img.shape
        landmark_list = []
        
        # Use results if provided, otherwise process the image
        if results is None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # If the requested hand exists in the detected hands
            if hand_index < len(results.multi_hand_landmarks):
                my_hand = results.multi_hand_landmarks[hand_index]
                
                # Extract coordinates for each landmark
                for id, landmark in enumerate(my_hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
        
        return landmark_list

    def recognize_sign(self, landmark_list):
        """
        Recognize hand sign based on landmark positions.
        
        Parameters:
            landmark_list (list): List of landmark positions
            
        Returns:
            sign (str): Recognized hand sign
            confidence (float): Confidence level of the recognition
        """
        # If no landmarks are detected, return 'Unknown'
        if not landmark_list:
            return "Unknown", 0.0
            
        # Dictionary to store detected signs and their confidence
        signs = {}
        
        # Extract important finger tips and bases
        # MediaPipe Hand Landmarks:
        # - Wrist: 0
        # - Thumb: 1 (CMC) -> 2 (MCP) -> 3 (IP) -> 4 (TIP)
        # - Index: 5 (MCP) -> 6 -> 7 -> 8 (TIP)
        # - Middle: 9 (MCP) -> 10 -> 11 -> 12 (TIP)
        # - Ring: 13 (MCP) -> 14 -> 15 -> 16 (TIP)
        # - Pinky: 17 (MCP) -> 18 -> 19 -> 20 (TIP)
        
        # Check if we have all required landmarks
        if len(landmark_list) < 21:
            return "Unknown", 0.0
            
        # Get key landmark positions
        wrist = landmark_list[0][1:] if landmark_list[0] else None
        thumb_tip = landmark_list[4][1:] if landmark_list[4] else None
        index_tip = landmark_list[8][1:] if landmark_list[8] else None
        middle_tip = landmark_list[12][1:] if landmark_list[12] else None
        ring_tip = landmark_list[16][1:] if landmark_list[16] else None
        pinky_tip = landmark_list[20][1:] if landmark_list[20] else None
        
        thumb_base = landmark_list[2][1:] if landmark_list[2] else None
        index_base = landmark_list[5][1:] if landmark_list[5] else None
        middle_base = landmark_list[9][1:] if landmark_list[9] else None
        ring_base = landmark_list[13][1:] if landmark_list[13] else None
        pinky_base = landmark_list[17][1:] if landmark_list[17] else None
        
        # Helper function to calculate distance between two points
        def calculate_distance(p1, p2):
            if p1 and p2:
                return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            return 0
            
        # Helper function to check if a finger is extended
        def is_finger_extended(tip, base, wrist, threshold=1.0):
            if tip and base and wrist:
                # Distance from tip to wrist should be greater than distance from base to wrist
                tip_to_wrist = calculate_distance(tip, wrist)
                base_to_wrist = calculate_distance(base, wrist)
                return tip_to_wrist > (base_to_wrist * threshold)
            return False
            
        # Check finger states (extended or not)
        is_thumb_up = is_finger_extended(thumb_tip, thumb_base, wrist, 1.1)
        is_index_up = is_finger_extended(index_tip, index_base, wrist)
        is_middle_up = is_finger_extended(middle_tip, middle_base, wrist)
        is_ring_up = is_finger_extended(ring_tip, ring_base, wrist)
        is_pinky_up = is_finger_extended(pinky_tip, pinky_base, wrist)
        
        # Check thumb to index pinch
        is_thumb_index_pinched = False
        if thumb_tip and index_tip:
            thumb_index_distance = calculate_distance(thumb_tip, index_tip)
            wrist_index_distance = calculate_distance(wrist, index_tip)
            if thumb_index_distance < (wrist_index_distance * 0.2):
                is_thumb_index_pinched = True
                
        # Number signs 0-5
        # Number 0 (fist with thumb wrapped)
        if (not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up and not is_thumb_up):
            signs["0"] = 0.8
            
        # Number 1 (index finger up, others down)
        if (is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up):
            signs["1"] = 0.9
            
        # Number 2 (index and middle fingers up, others down)
        if (is_index_up and is_middle_up and not is_ring_up and not is_pinky_up):
            signs["2"] = 0.9
            signs["Peace"] = 0.9  # Peace sign is the same as 2
            
        # Number 3 (index, middle, and ring fingers up, pinky down)
        if (is_index_up and is_middle_up and is_ring_up and not is_pinky_up):
            signs["3"] = 0.9
            
        # Number 4 (all fingers except thumb up)
        if (is_index_up and is_middle_up and is_ring_up and is_pinky_up and not is_thumb_up):
            signs["4"] = 0.9
            
        # Number 5 (all fingers up)
        if (is_index_up and is_middle_up and is_ring_up and is_pinky_up and is_thumb_up):
            signs["5"] = 0.9
            
        # Thumbs Up
        if (is_thumb_up and not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up):
            signs["Thumbs Up"] = 0.9
            
        # OK Sign (thumb and index form a circle, other fingers extended)
        if (is_thumb_index_pinched and is_middle_up and is_ring_up and is_pinky_up):
            signs["Okay"] = 0.9
            
        # Rock sign (index and pinky up, others down)
        if (is_index_up and not is_middle_up and not is_ring_up and is_pinky_up):
            signs["Rock"] = 0.9
            
        # Point (only index finger up)
        if (is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up and not is_thumb_up):
            signs["Point"] = 0.9
            
        # Find the sign with the highest confidence
        if signs:
            max_sign = max(signs, key=signs.get)
            return max_sign, signs[max_sign]
        else:
            return "Unknown", 0.0
            
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
            "Point": "point"
        }
        
        # Check if the sign is one we want to count and if the cooldown has passed
        current_time = time.time()
        if sign in gesture_map and current_time - self.last_gesture_time >= self.gesture_cooldown:
            self.gesture_counts[gesture_map[sign]] += 1
            self.last_gesture_time = current_time

def main():
    """
    Main function to run the hand sign detection application.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize detector
    detector = HandSignDetector(min_detection_confidence=0.7)
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break
            
        # Mirror the image horizontally for a more natural interaction
        img = cv2.flip(img, 1)
        
        # Find hands in the frame
        img, results = detector.find_hands(img)
        
        # Find position of landmarks
        landmark_list = detector.find_positions(img, results=results)
        
        # Recognize hand sign if landmarks were found
        sign, confidence = detector.recognize_sign(landmark_list)
        
        # Count gestures
        detector.count_gesture(sign)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        
        # Display information on the image
        # FPS counter
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display detected sign and confidence if a sign was detected
        if sign != "Unknown":
            cv2.putText(img, f'{sign} ({confidence:.2f})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
        
        # Display gesture counts
        y_offset = 110
        for gesture, count in detector.gesture_counts.items():
            cv2.putText(img, f'{gesture}: {count}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        # Display the image
        cv2.imshow('Hand Sign Detection', img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()