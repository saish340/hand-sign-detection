# Hand Sign Detection

A real-time hand sign detection application built with Python, OpenCV, and MediaPipe.

## Features

- Real-time hand detection and tracking using webcam
- Recognition of various hand signs and gestures:
  - Number signs (0-5)
  - Thumbs up gesture
  - Peace sign
  - Okay sign
  - Rock sign
  - Pointing gesture
- Gesture counting and tracking
- FPS display
- Clean object-oriented code structure

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hand-sign-detection.git
   cd hand-sign-detection
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe numpy
   ```

## Usage

Run the main application:
```
python hand_sign_detector.py
```

View the hand sign reference guide:
```
python hand_sign_reference.py
```

Press 'q' to exit the application.

## How It Works

1. The application captures video from your webcam.
2. MediaPipe's Hands solution detects and tracks hand landmarks (21 points per hand).
3. The custom algorithm analyzes the relationship between these landmarks to recognize specific hand signs.
4. The detected hand sign is displayed on the video feed along with confidence level.
5. Special gestures (like thumbs up) are counted with a cooldown to avoid multiple counts.

## Customization

You can modify the `HandSignDetector` class in `hand_sign_detector.py` to:
- Add new hand signs by updating the `recognize_sign` method
- Adjust detection thresholds for better accuracy
- Track additional gestures by updating the `gesture_counts` dictionary

## License

[MIT License](LICENSE)

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand detection solution
- [OpenCV](https://opencv.org/) for the computer vision framework