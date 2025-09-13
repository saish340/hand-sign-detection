"""
Basic Hand Detection using OpenCV
--------------------------------
This is a simplified version of the hand sign detector that uses only OpenCV
without MediaPipe for environments where MediaPipe cannot be installed.

Author: GitHub Copilot
Date: September 13, 2025
"""

import cv2
import numpy as np
import time

def main():
    """
    Main function to run the basic hand detection application.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    # Create a background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    print("Basic Hand Detection started.")
    print("Press 'b' to capture background.")
    print("Press 'q' to quit.")
    
    # For skin color detection
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    background_captured = False
    
    while True:
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break
            
        # Mirror the image horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Apply background subtraction or skin detection based on mode
        if background_captured:
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            
            # Apply some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Convert to HSV color space for skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask for skin color detection
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour which should be the hand
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough (hand, not noise)
            if cv2.contourArea(max_contour) > 1000:
                # Get the bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Draw the bounding rectangle around the hand
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw the contour
                cv2.drawContours(display_frame, [max_contour], 0, (0, 0, 255), 2)
                
                # Get the convex hull of the contour
                hull = cv2.convexHull(max_contour, returnPoints=False)
                
                # Find convexity defects
                try:
                    defects = cv2.convexityDefects(max_contour, hull)
                    
                    # Initialize finger count
                    finger_count = 0
                    
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
                                
                                # Draw the defect point
                                cv2.circle(display_frame, far, 5, [255, 0, 0], -1)
                        
                        # Add 1 for the thumb
                        finger_count = min(finger_count + 1, 5)
                        
                        # Display the finger count
                        cv2.putText(display_frame, f'Fingers: {finger_count}', 
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except:
                    # Just draw the contour if convexity defects fail
                    pass
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        
        # Display FPS on the image
        cv2.putText(display_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display the mode
        mode_text = "BG Subtraction" if background_captured else "Skin Detection"
        cv2.putText(display_frame, mode_text, (10, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
        # Display the image
        cv2.imshow('Basic Hand Detection', display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit if 'q' is pressed
            break
        elif key == ord('b'):
            # Capture background if 'b' is pressed
            background_captured = True
            print("Background captured. Using background subtraction method.")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()