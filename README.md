# Digital_Canvas

![image](https://github.com/user-attachments/assets/436f53dc-0371-4c20-8474-c2298dce3406)
![image](https://github.com/user-attachments/assets/4c299a01-a0c6-42ec-b423-834e6a04da7e)

Install the following libraries by running
```bash
pip install opencv-python numpy mediapipe tensorflow PyQt5
sudo apt update
sudo apt install -y libgl1
```
 Python 3.11 latest
 ```
Press Command + Shift + P (on macOS) to open the Command Palette.
Type: "Python: Select Interpreter" and hit enter.
In the list that appears, choose the interpreter:
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
This will update VS Code’s settings so the run button uses that Python. No need for export PATH=..., as VS Code will now directly call the correct interpreter.

Once selected, try running your file again. It should now run without the “no such file or directory” error.
```
Adjust Your PATH (Optional): If you prefer just typing python3 instead of the full path, adjust your PATH so that the framework’s Python comes first. For example:
```
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
python3 air_canvas_hand.py
```
Requirements:
```bash
Python 3.x
OpenCV
NumPy
MediaPipe (for hand recognition)
TensorFlow (for object detection)
PyQt5.QtWidgets (for image saving to disk and UI/UX)
A webcam or camera connected to your computer
```

For running Air Canvas with Hand Recognition, 
run
```bash
python3 air_canvas_hand.py
```
Create a new branch
``` When editing and changes
Please create you own individual branch so we can review and merge into main.
```
Current Hand Gestures:
```
Index finger: Drawing/Writing mode
Index and Middle finger: Selection mode
Three fingers with index, middle, and ring fingers: Go back to the previous tool or color
Four fingers with thumb down: Eraser mode
Thumb with index finger (pinch): Pause temporary
Pinky to thumb (lasso): Select an area to transform
5 finger (transformation): To scale, enlarge or minimalize canvas or lasso
```
