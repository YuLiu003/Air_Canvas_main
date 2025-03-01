# Digital Canvas

An interactive air canvas application that uses computer vision to create digital art through hand gestures.

![Digital Canvas Demo 1](https://github.com/user-attachments/assets/436f53dc-0371-4c20-8474-c2298dce3406)
![Digital Canvas Demo 2](https://github.com/user-attachments/assets/4c299a01-a0c6-42ec-b423-834e6a04da7e)

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- MediaPipe (for hand recognition)
- TensorFlow (for object detection)
- PyQt5 (for UI/UX and saving images)
- Webcam or camera connected to your computer

## Installation

Install the required libraries:

```bash
pip install opencv-python numpy mediapipe tensorflow PyQt5
sudo apt update
sudo apt install -y libgl1
```

## Running the Application

To start the air canvas with hand recognition:

```bash
python3 air_canvas_hand.py
```

## Python Interpreter Setup

### VS Code Setup

1. Press Command + Shift + P (on macOS) to open the Command Palette
2. Type "Python: Select Interpreter" and hit enter
3. Choose the interpreter: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`

This configures VS Code to use the correct Python interpreter directly.

### Alternative: Adjust Your PATH

If you prefer using `python3` command without the full path:

```bash
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
python3 air_canvas_hand.py
```

## Contributing

When making changes:

1. Create your fork
2. Make your modifications
3. Submit for review and merging into main

## Hand Gestures Guide

| Gesture | Function |
|---------|----------|
| Index finger only | Drawing/Writing mode |
| Index and Middle fingers | Selection mode |
| Three fingers (index, middle, ring) | Go back to previous tool/color |
| Four fingers with thumb down | Eraser mode |
| Thumb and index finger (pinch) | Pause temporarily |
| Pinky to thumb (lasso) | Select an area to transform |
| All five fingers | Transform canvas/selection (scale, enlarge, minimize) |
