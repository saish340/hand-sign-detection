"""
Hand Sign Reference Guide
------------------------
This file provides information about the hand signs that can be recognized
by the hand_sign_detector.py application.

Author: GitHub Copilot
Date: September 13, 2025
"""

HAND_SIGNS = {
    # Number signs
    "0": "Make a fist with all fingers closed",
    "1": "Extend only the index finger",
    "2": "Extend index and middle fingers (peace sign)",
    "3": "Extend index, middle, and ring fingers",
    "4": "Extend index, middle, ring, and pinky fingers (thumb closed)",
    "5": "Open hand with all fingers extended",
    
    # Gesture signs
    "Thumbs Up": "Make a fist and extend only the thumb upward",
    "Peace": "Extend index and middle fingers in a V shape",
    "Okay": "Form a circle with thumb and index finger, extend other fingers",
    "Rock": "Extend index finger and pinky, keep middle and ring fingers closed",
    "Point": "Extend only the index finger (same as number 1)"
}

def print_hand_signs_guide():
    """Print information about all the recognizable hand signs"""
    print("\n" + "="*50)
    print(" "*15 + "HAND SIGN REFERENCE GUIDE")
    print("="*50)
    
    print("\nNUMBER SIGNS:")
    for i in range(6):  # 0-5
        print(f"  {i}: {HAND_SIGNS[str(i)]}")
    
    print("\nGESTURE SIGNS:")
    gestures = ["Thumbs Up", "Peace", "Okay", "Rock", "Point"]
    for gesture in gestures:
        print(f"  {gesture}: {HAND_SIGNS[gesture]}")
    
    print("\n" + "="*50)
    print("To use the hand sign detector, run: python hand_sign_detector.py")
    print("Press 'q' to quit the application")
    print("="*50 + "\n")

if __name__ == "__main__":
    print_hand_signs_guide()