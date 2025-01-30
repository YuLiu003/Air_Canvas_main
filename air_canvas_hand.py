# air_canvas_hand.py

import cv2
import numpy as np
import os
import sys
import math
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from air_canvas_instructions import show_instructions
import HandTrack as htp

# Set fixed dimensions for the window
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 780
HEADER_HEIGHT = 135

# Set different tool thicknesses
PEN_THICKNESS = 6
BRUSH_THICKNESS = 24
MARKER_THICKNESS = 24
ERASER_THICKNESS = 50

# Set indicator sizes (base sizes, will adjust based on finger positions)
CIRCLE_RADIUS = 15
MIN_RECTANGLE_SIZE = 30
MIN_ERASER_SIZE = 40

# Add these constants at the top
MIN_POINT_DISTANCE = 10  # Minimum pixels between points
SMOOTHING_FACTOR = 0.5  # Point smoothing factor
LASSO_COLOR = (0, 255, 0)  # Green color for lasso preview
LASSO_THICKNESS = 2

# Add new constants
TRANSFORM_SMOOTHING = 0.3  # Smoothing factor for transformations
MASK_BLUR_SIZE = (3, 3)   # Size of Gaussian blur for mask edges
MASK_BLUR_SIGMA = 1.5     # Sigma for Gaussian blur

# Create a blank canvas to draw on
imgCanvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
imgMarkerCanvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)

# Load overlay images in the correct order

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Update resource paths
folderPath = get_resource_path("Header")

imPaths = [imPath for imPath in os.listdir(folderPath) if imPath.endswith('.jpg')]
imPaths.sort(key=lambda f: int(os.path.splitext(f)[0]))  # Sort numerically by filename

overlayList = [cv2.imread(os.path.join(folderPath, imPath)) for imPath in imPaths]

# Set default header to the first image corresponding to 'Red'
if overlayList:
    header = overlayList[0]  # Assuming 'Red' is the first in overlayList
else:
    header = np.zeros((HEADER_HEIGHT, WINDOW_WIDTH, 3), np.uint8)  # Placeholder header

new_header = True

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

# Initialize the hand detector
detector = htp.handDetector(detectionCon=0.85)

# Define a Command class
class Command:
    def __init__(self, action, **kwargs):
        self.action = action  # 'draw' or 'clear'
        self.kwargs = kwargs  # Additional arguments needed for the action

    def execute(self, canvas, marker_canvas):
        if self.action == 'draw':
            start_point = self.kwargs['start_point']
            end_point = self.kwargs['end_point']
            color = self.kwargs['color']
            thickness = self.kwargs['thickness']
            tool = self.kwargs['tool']
            if tool == "Marker":
                cv2.line(marker_canvas, start_point, end_point, color, thickness)
            elif tool == "Eraser":
                # Erase from both canvases by drawing black lines
                cv2.line(canvas, start_point, end_point, (0, 0, 0), thickness)
                cv2.line(marker_canvas, start_point, end_point, (0, 0, 0), thickness)
            else:
                cv2.line(canvas, start_point, end_point, color, thickness)
        elif self.action == 'clear':
            canvas.fill(0)
            marker_canvas.fill(0)

# Canvas state management class
class CanvasState:
    def __init__(self):
        self.xp, self.yp = 0, 0
        self.drawing = False
        self.first_frame = True
        self.current_mode = "Selection"
        self.drawColor = (0, 0, 255)  # Default to red in BGR format
        self.previous_color = self.drawColor
        self.last_non_eraser_color = self.drawColor
        self.current_thickness = PEN_THICKNESS
        self.current_tool = "Pen"  # Add current tool tracking
        self.previous_tool = "Pen"  # Add previous tool tracking
        self.previous_thickness = PEN_THICKNESS  # Add previous thickness tracking
        self.is_paused = False  # New state variable for pause functionality
        # Add state variables for edit mode
        self.is_edit_mode = False
        self.edit_start_pos = None
        self.initial_canvas = None
        self.initial_marker_canvas = None  # **Add this line**
        self.scale_factor = 1.0
        # Store complete tool state
        self.tool_states = {
            "Pen": {"thickness": PEN_THICKNESS, "color": (0, 0, 255)},
            "Brush": {"thickness": BRUSH_THICKNESS, "color": (0, 0, 255)},
            "Marker": {"thickness": MARKER_THICKNESS, "color": (0, 255, 255)}
        }
        self.commands = []
        self.command_index = -1  # Points to the last executed command
        self.checkpoints = {}  # For optimizing redraws
        self.checkpoint_interval = 50  # Save a checkpoint every 50 commands
        self.lasso_points = []
        self.is_lasso_mode = False
        self.lasso_mask = None
        self.lasso_selection = None
        self.initial_lasso_pos = None
        self.has_active_selection = False  # New flag to track active selection
        self.previous_canvas = None  # Store canvas state before transformation
        self.selection_background = None  # Store background under selection
        self.move_completed = False  # Track if move is completed
        self.last_transform = None  # Store last transform state
        self.original_canvas = None  # Store complete canvas before selection
        self.transform_buffer = None  # Buffer for smooth transformations
        self.last_delta = None       # Store last transformation delta
        self.last_scale = 1.0        # Store last scale factor
        self.edit_start_pos = None
        self.edit_initial_distance = None
        self.lasso_start_pos = None
        self.lasso_initial_distance = None
        self.original_marker_canvas = None  # Add this line
        self.lasso_selection_marker = None  # Add this line
        self.selection_background_marker = None  # Add this line

state = CanvasState()

def move_canvas(canvas, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved_canvas = cv2.warpAffine(canvas, M, (WINDOW_WIDTH, WINDOW_HEIGHT))
    return moved_canvas

def resize_canvas(canvas, center, scale_factor):
    h, w = canvas.shape[:2]
    # Get the rotation matrix for scaling
    M = cv2.getRotationMatrix2D(center, 0, scale_factor)
    resized_canvas = cv2.warpAffine(canvas, M, (w, h))
    return resized_canvas

# Function to move the canvas
def move_canvases(canvases, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved_canvases = [cv2.warpAffine(canvas, M, (WINDOW_WIDTH, WINDOW_HEIGHT)) for canvas in canvases]
    return moved_canvases

# Function to resize the canvas
def resize_canvases(canvases, center, scale_factor):
    h, w = canvases[0].shape[:2]
    # Get the rotation matrix for scaling
    M = cv2.getRotationMatrix2D(center, 0, scale_factor)
    resized_canvases = [cv2.warpAffine(canvas, M, (w, h)) for canvas in canvases]
    return resized_canvases

# Function to draw mode indicators
def calculate_finger_center(lmlist, finger1_id, finger2_id):
    """Calculate the center point between two fingers"""
    x1, y1 = lmlist[finger1_id][1:]
    x2, y2 = lmlist[finger2_id][1:]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y, abs(x2 - x1)

def draw_mode_indicator(img, lmlist, mode, color):
    if mode == "Drawing":
        # Draw a circle for drawing mode at index finger tip
        x1, y1 = lmlist[8][1:]  # Index finger tip
        cv2.circle(img, (x1, y1), CIRCLE_RADIUS, color, -1)
        cv2.circle(img, (x1, y1), CIRCLE_RADIUS, (255, 255, 255), 2)
    
    elif mode == "Selection":
        # Draw rectangle between index and middle finger
        if len(lmlist) >= 12:
            index_x, index_y = lmlist[8][1:]
            middle_x, middle_y = lmlist[12][1:]
            
            center_x = (index_x + middle_x) // 2
            center_y = (index_y + middle_y) // 2
            width = max(MIN_RECTANGLE_SIZE, abs(middle_x - index_x))
            height = width * 0.6
            
            x_rect = center_x - width // 2
            y_rect = center_y - height // 2
            cv2.rectangle(
                img, 
                (int(x_rect), int(y_rect)), 
                (int(x_rect + width), int(y_rect + height)), 
                color, 
                2
            )
    elif mode == "Edit":
        # Draw a circle around the hand to indicate Edit Mode
        center_x = lmlist[9][1]
        center_y = lmlist[9][2]
        hand_span = calculate_hand_span(lmlist)
        radius = int(hand_span / 2)
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 255), 2)
                
    elif mode == "Erasing":
        # Draw square between all four fingers when they're up
        if len(lmlist) >= 20:  # Make sure we have all finger positions
            # Get index and pinky finger tips for width calculation
            index_x, index_y = lmlist[8][1:]
            middle_x, middle_y = lmlist[12][1:]
            ring_x, ring_y = lmlist[16][1:]
            pinky_x, pinky_y = lmlist[20][1:]
            
            # Calculate center point of all four fingers
            center_x = (index_x + middle_x + ring_x + pinky_x) // 4
            center_y = (index_y + middle_y + ring_y + pinky_y) // 4
            
            # Calculate size based on distance between index and pinky
            size = max(MIN_ERASER_SIZE, abs(pinky_x - index_x))
            
            # Draw square centered between fingers
            x_square = center_x - size // 2
            y_square = center_y - size // 2
            
            # Draw the eraser square
            cv2.rectangle(img, 
                         (int(x_square), int(y_square)),
                         (int(x_square + size), int(y_square + size)),
                         (0, 0, 0), 2)
            
            # Add cross pattern inside eraser square
            cv2.line(img, 
                    (int(x_square), int(y_square)), 
                    (int(x_square + size), int(y_square + size)), 
                    (0, 0, 0), 2)
            cv2.line(img, 
                    (int(x_square + size), int(y_square)),
                    (int(x_square), int(y_square + size)), 
                    (0, 0, 0), 2)
        
# Function to handle selection mode with fixed regions
def handle_selection_mode(x1, y1):
    global header, new_header

    # Define selection areas with fixed positions (x_start, x_end)
    selection_areas = {
        "Red": (20, 80),
        "Orange": (100, 160),
        "Yellow": (180, 240),
        "Green": (260, 320),
        "Blue": (340, 400),
        "Purple": (420, 480),
        "Pink": (500, 560),
        "Black": (580, 640),
        "White": (660, 720),
        "Eraser": (740, 800),
        "Pen": (820, 880),
        "Brush": (900, 960),
        "Marker": (980, 1040),
        "Undo": (1060, 1120),
        "Redo": (1140, 1200),
        "Save": (1220, 1280),
        "Clear": (1300, 1360),
    }

    # Define colors in BGR format
    color_values = {
        "Red": (0, 0, 255),
        "Orange": (0, 165, 255),
        "Yellow": (0, 255, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Purple": (128, 0, 128),
        "Pink": (147, 20, 255),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Eraser": (0, 0, 0)
    }

    colors_in_order = list(selection_areas.keys())

    # Only process selection if finger is in the header area
    if y1 < HEADER_HEIGHT:
        for action, (x_min, x_max) in selection_areas.items():
            if x_min < x1 < x_max:
                header_index = list(selection_areas.keys()).index(action)
                if header_index < len(overlayList):
                    header = overlayList[header_index]
                else:
                    header = np.zeros((HEADER_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
                
                # Perform actions based on the selection
                if action == "Save":
                    save_drawing()
                elif action == "Clear":
                    clear_canvas()
                elif action == "Undo":
                    if state.command_index >= 0:
                        state.command_index -= 1
                        redraw_canvas()
                elif action == "Redo":
                    if state.command_index < len(state.commands) - 1:
                        state.command_index += 1
                        state.commands[state.command_index].execute(imgCanvas, imgMarkerCanvas)

                elif action == "Pen":
                    state.current_tool = "Pen"
                    state.current_thickness = PEN_THICKNESS
                    # Restore the last color used with pen
                    state.drawColor = state.tool_states["Pen"]["color"]
                elif action == "Brush":
                    state.current_tool = "Brush"
                    state.current_thickness = BRUSH_THICKNESS
                    # Restore the last color used with brush
                    state.drawColor = state.tool_states["Brush"]["color"]
                elif action == "Marker":
                    state.current_tool = "Marker"
                    state.current_thickness = MARKER_THICKNESS
                    state.drawColor = state.tool_states["Marker"]["color"]
                elif action == "Eraser":
                    # Store the current tool state before switching to eraser
                    if state.current_tool in state.tool_states:
                        state.tool_states[state.current_tool]["color"] = state.drawColor
                        state.tool_states[state.current_tool]["thickness"] = state.current_thickness
                    state.previous_tool = state.current_tool
                    state.current_tool = "Eraser"
                    state.drawColor = (0, 0, 0)
                    state.current_thickness = ERASER_THICKNESS
                else:
                    # Update color for current tool
                    new_color = color_values.get(action, (0, 0, 0))
                    if new_color != (0, 0, 0):  # If not eraser color
                        state.drawColor = new_color
                        if state.current_tool in state.tool_states:
                            state.tool_states[state.current_tool]["color"] = new_color
                
                new_header = True
                break

# Function to draw on the canvas
def draw_on_canvas(imgCanvas, imgMarkerCanvas, x1, y1, xp, yp, drawColor, thickness):
    command = Command(
        action='draw',
        start_point=(xp, yp),
        end_point=(x1, y1),
        color=drawColor,
        thickness=thickness,
        tool=state.current_tool
    )    # Discard any commands beyond the current index
    if state.command_index < len(state.commands) - 1:
        state.commands = state.commands[:state.command_index+1]
        # Also remove any checkpoints beyond the current index
        state.checkpoints = {k: v for k, v in state.checkpoints.items() if k <= state.command_index}
    # Append the new command
    state.commands.append(command)
    state.command_index += 1
    command.execute(imgCanvas, imgMarkerCanvas)
    # Save checkpoint if needed
    if state.command_index % state.checkpoint_interval == 0:
        state.checkpoints[state.command_index] = (imgCanvas.copy(), imgMarkerCanvas.copy())

def calculate_hand_span(lmlist):
    thumb_tip = lmlist[4][1:]
    pinky_tip = lmlist[20][1:]
    distance = math.hypot(pinky_tip[0] - thumb_tip[0], thumb_tip[1] - pinky_tip[1])
    return distance

# Function to rotate an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Function to save the drawing
def save_drawing():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getSaveFileName(
        None, "Save your drawing", "", "PNG files (*.png);;All Files (*)")
    if file_path:
        # Combine both canvases appropriately before saving
        combined_canvas = imgCanvas.copy()
        cv2.addWeighted(combined_canvas, 1, imgMarkerCanvas, 0.3, 0, combined_canvas)
        cv2.imwrite(file_path, combined_canvas)
        print(f"Drawing saved to {file_path}")
    app.exit()

# Function to clear the canvas with confirmation
def clear_canvas(force=False):
    global imgCanvas, imgMarkerCanvas
    if not force:
        app = QApplication(sys.argv)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Clearing Canvas")
        msg_box.setText("Are you sure you want to clear the canvas?")
        msg_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        
        # Change button texts
        clear_button = msg_box.button(QMessageBox.Ok)
        clear_button.setText("Clear")
        cancel_button = msg_box.button(QMessageBox.Cancel)
        cancel_button.setText("Cancel")
        
        # Apply styles to make buttons blue, rounded, and spaced apart
        msg_box.setStyleSheet("""
            QPushButton {
                background-color: blue;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                margin: 0 20px;
            }
            QPushButton:hover {
                background-color: #0000aa;
            }
            QMessageBox {
                background-color: black;
            }
        """)
        ret = msg_box.exec_()
        if ret == QMessageBox.Ok:
            # Record the clear action
            command = Command(action='clear')
            # Discard any commands beyond the current index
            if state.command_index < len(state.commands) - 1:
                state.commands = state.commands[:state.command_index+1]
                state.checkpoints = {k: v for k, v in state.checkpoints.items() if k <= state.command_index}
            # Append the new command
            state.commands.append(command)
            state.command_index += 1
            # Execute the clear action
            command.execute(imgCanvas, imgMarkerCanvas)
            # Save checkpoint if needed
            if state.command_index % state.checkpoint_interval == 0:
                state.checkpoints[state.command_index] = (imgCanvas.copy(), imgMarkerCanvas.copy())
        app.exit()
    else:
        imgCanvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
        imgMarkerCanvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)

# Redraw canvas from commands
def redraw_canvas():
    # Find the nearest checkpoint
    checkpoint_indices = [i for i in state.checkpoints.keys() if i <= state.command_index]
    if checkpoint_indices:
        last_checkpoint_index = max(checkpoint_indices)
        imgCanvas[:], imgMarkerCanvas[:] = state.checkpoints[last_checkpoint_index]
        start_index = last_checkpoint_index + 1
    else:
        imgCanvas[:] = 0  # Clear the canvas
        imgMarkerCanvas[:] = 0
        start_index = 0
    for i in range(start_index, state.command_index + 1):
        state.commands[i].execute(imgCanvas, imgMarkerCanvas)

# Add new helper function
def reset_selection_state(state, canvas, marker_canvas):
    """Reset all selection-related states and finalize any active selection"""
    if state.has_active_selection and not state.is_lasso_mode:
        if state.selection_background is not None:
            # Recreate the transformation matrix
            cx, cy = state.initial_lasso_pos
            dx, dy = state.last_delta if state.last_delta is not None else (0, 0)
            s = state.last_scale if state.last_scale is not None else 1.0

            M = np.float32([
                [s, 0, (1 - s) * cx + dx],
                [0, s, (1 - s) * cy + dy]
            ])

            # Transform both regular and marker selections
            transformed = cv2.warpAffine(state.lasso_selection, M, (WINDOW_WIDTH, WINDOW_HEIGHT))
            transformed_marker = cv2.warpAffine(
                state.lasso_selection_marker, M, (WINDOW_WIDTH, WINDOW_HEIGHT),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            transformed_mask = cv2.warpAffine(state.lasso_mask, M, (WINDOW_WIDTH, WINDOW_HEIGHT))
            transformed_mask = cv2.GaussianBlur(transformed_mask, MASK_BLUR_SIZE, MASK_BLUR_SIGMA)
            
            # Create inverted mask
            mask_inv = cv2.bitwise_not(transformed_mask)
            
            # Blend regular canvas
            temp_canvas = state.selection_background.copy()
            cv2.add(temp_canvas, transformed, temp_canvas, mask=transformed_mask)
            canvas[:] = temp_canvas

            # Blend marker canvas - preserve existing marker strokes
            temp_marker_canvas = state.selection_background_marker.copy()
            cv2.add(temp_marker_canvas, transformed_marker, temp_marker_canvas, mask=transformed_mask)
            marker_canvas[:] = temp_marker_canvas

    # Reset all selection states
    state.lasso_points = []
    state.lasso_selection = None
    state.lasso_selection_marker = None
    state.lasso_mask = None
    state.has_active_selection = False
    state.initial_lasso_pos = None
    state.transform_buffer = None
    state.last_delta = None
    state.last_scale = 1.0
    state.previous_canvas = None
    state.selection_background = None
    state.selection_background_marker = None
    state.move_completed = False
    state.original_canvas = None
    state.original_marker_canvas = None

    return canvas, marker_canvas

# Add new helper function for point smoothing
def smooth_point(prev_point, new_point, smoothing_factor):
    if prev_point is None:
        return new_point
    return (
        int(prev_point[0] * smoothing_factor + new_point[0] * (1 - smoothing_factor)),
        int(prev_point[1] * smoothing_factor + new_point[1] * (1 - smoothing_factor))
    )

# Add distance check function
def point_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p1[1] - p2[1])**2)

# Add new helper function for smooth transformations
def smooth_transform(current, target, smoothing_factor):
    if current is None:
        return target
    return current * smoothing_factor + target * (1 - smoothing_factor)

show_instructions()

# Main loop for real-time hand detection and drawing
while True:
    success, img = cap.read()
    if not success:
        break

    # Resize the captured frame to fixed dimensions
    img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    img = cv2.flip(img, 1)

    # Detect hands in the frame
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # Index finger tip

        # Draw mode indicator based on current mode and all finger positions
        if not state.is_paused:
            draw_mode_indicator(img, lmlist, state.current_mode, state.drawColor)

        gesture = detector.detectGesture()

        if gesture == "FiveFingers":
            if state.lasso_selection is not None and state.has_active_selection:
                # Lasso transformation
                center_x = lmlist[9][1]
                center_y = lmlist[9][2]
                current_pos = np.array([center_x, center_y])

                if state.lasso_start_pos is None:
                    state.lasso_start_pos = current_pos
                    state.lasso_initial_distance = calculate_hand_span(lmlist)
                else:
                    # Calculate movement with smoothing
                    delta_pos = current_pos - state.lasso_start_pos
                    if state.last_delta is not None:
                        delta_pos = smooth_transform(state.last_delta, delta_pos, TRANSFORM_SMOOTHING)
                    state.last_delta = delta_pos

                    # Create transformation buffer if needed
                    if state.transform_buffer is None:
                        state.transform_buffer = state.lasso_selection.copy()

                    # Move selection with smooth blending
                    transformed = move_canvas(state.lasso_selection,
                                              int(delta_pos[0]),
                                              int(delta_pos[1]))

                    # Calculate scaling with smoothing
                    current_distance = calculate_hand_span(lmlist)
                    if state.lasso_initial_distance != 0:
                        scale = current_distance / state.lasso_initial_distance
                        scale = smooth_transform(state.last_scale, scale, TRANSFORM_SMOOTHING)
                        state.last_scale = scale

                        transformed = resize_canvas(transformed,
                                                    (center_x, center_y),
                                                    scale)

                    # Update transform buffer
                    state.transform_buffer = transformed.copy()

                    # Create transformed mask
                    transformed_mask = move_canvas(state.lasso_mask,
                                                   int(delta_pos[0]),
                                                   int(delta_pos[1]))
                    transformed_mask = cv2.GaussianBlur(transformed_mask,
                                                        MASK_BLUR_SIZE,
                                                        MASK_BLUR_SIGMA)

                    # Blend with background using double buffering
                    temp_canvas = state.selection_background.copy()
                    cv2.add(temp_canvas, transformed, temp_canvas,
                            mask=transformed_mask)
                    
                    temp_marker_canvas = state.selection_background_marker.copy()
                    cv2.add(temp_marker_canvas, transformed_marker, temp_marker_canvas, mask=transformed_mask)

                    # Update canvas with minimal flickering
                    imgCanvas[:] = temp_canvas
                    imgMarkerCanvas[:] = temp_marker_canvas

            else:
                # Whole canvas transformation
                state.is_edit_mode = True
                state.current_mode = "Edit"

                # Use the center of the hand for transformations
                center_x = lmlist[9][1]
                center_y = lmlist[9][2]
                current_pos = np.array([center_x, center_y])

                if state.edit_start_pos is None:
                    state.edit_start_pos = current_pos
                    state.initial_canvas = imgCanvas.copy()
                    state.initial_marker_canvas = imgMarkerCanvas.copy()
                    state.edit_initial_distance = calculate_hand_span(lmlist)
                else:
                    # Calculate movement
                    delta_pos = current_pos - state.edit_start_pos
                    imgCanvas, imgMarkerCanvas = move_canvases(
                        [state.initial_canvas, state.initial_marker_canvas],
                        delta_pos[0],
                        delta_pos[1]
                    )
                    # Calculate scaling
                    current_distance = calculate_hand_span(lmlist)
                    if state.edit_initial_distance != 0:
                        scale_factor = current_distance / state.edit_initial_distance
                        # Resize both canvases
                        imgCanvas, imgMarkerCanvas = resize_canvases(
                            [imgCanvas, imgMarkerCanvas],
                            (center_x, center_y),
                            scale_factor
                        )
        else:
            # Reset edit mode state variables when not in "FiveFingers" gesture
            state.is_edit_mode = False
            state.edit_start_pos = None
            state.initial_canvas = None
            state.initial_marker_canvas = None  # Reset this as well
            state.edit_initial_distance = None

            # Also reset lasso transformation variables if not transforming lasso
            if not (gesture == "FiveFingers" and state.has_active_selection):
                state.lasso_start_pos = None
                state.lasso_initial_distance = None

        if gesture == "Pinch":
            # Only pause/unpause drawing actions
            state.is_paused = True
            # Show a subtle pause indicator
            cv2.putText(img, "⏸️", (WINDOW_WIDTH - 50, WINDOW_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            state.is_paused = False

        # Only process other gestures if not paused
        if not state.is_paused:

            if gesture == "ThreeFingers":
                # Three fingers up - Return to previous tool and color
                if state.current_tool == "Eraser":
                    # Restore the previous tool and its properties
                    state.current_tool = state.previous_tool
                    # Restore the tool's last used properties
                    if state.current_tool in state.tool_states:
                        state.drawColor = state.tool_states[state.current_tool]["color"]
                        state.current_thickness = state.tool_states[state.current_tool]["thickness"]
                state.current_mode = "Drawing"
                state.drawing = False
                state.first_frame = True

            elif gesture == "FourFingers":
                # Erasing mode: User shows four fingers
                if state.current_tool != "Eraser":
                    # Store the current tool state before switching to eraser
                    if state.current_tool in state.tool_states:
                        state.tool_states[state.current_tool]["color"] = state.drawColor
                        state.tool_states[state.current_tool]["thickness"] = state.current_thickness
                    state.previous_tool = state.current_tool
                state.current_mode = "Erasing"
                state.current_tool = "Eraser"
                state.drawColor = (0, 0, 0)
                state.current_thickness = ERASER_THICKNESS
                state.xp, state.yp = 0, 0
                state.drawing = False
                state.first_frame = True

            elif gesture == "TwoFingers":
                # Selection mode: Only index and middle fingers up
                state.current_mode = "Selection"
                state.xp, state.yp = 0, 0
                state.drawing = False
                state.first_frame = True
                handle_selection_mode(x1, y1)

            elif gesture == "OneFinger":
                # Drawing mode: Only index finger up
                state.current_mode = "Drawing"
                thickness = state.current_thickness

                if state.first_frame:
                    state.xp, state.yp = x1, y1
                    state.first_frame = False
                else:
                    draw_on_canvas(imgCanvas, imgMarkerCanvas, x1, y1, state.xp, state.yp, state.drawColor, thickness)
                    state.xp, state.yp = x1, y1
                state.drawing = True

            elif gesture == "Lasso":
                if state.has_active_selection and not state.is_lasso_mode:
                    imgCanvas, imgMarkerCanvas = reset_selection_state(state, imgCanvas, imgMarkerCanvas)
                if not state.is_lasso_mode:
                    state.is_lasso_mode = True
                    state.current_mode = "Lasso"
                    state.original_canvas = imgCanvas.copy()
                    state.original_marker_canvas = imgMarkerCanvas.copy()
                    state.lasso_points = []

                # Get current point
                x1, y1 = lmlist[8][1:]  # Index finger tip
                current_point = (x1, y1)

                # Only add point if it's far enough from last point
                if not state.lasso_points or point_distance(state.lasso_points[-1], current_point) > MIN_POINT_DISTANCE:
                    # Smooth point based on previous point
                    smoothed_point = smooth_point(
                        state.lasso_points[-1] if state.lasso_points else None,
                        current_point,
                        SMOOTHING_FACTOR
                    )
                    state.lasso_points.append(smoothed_point)

                # Draw lasso preview with anti-aliasing
                if len(state.lasso_points) > 1:
                    # Convert points to numpy array
                    pts = np.array(state.lasso_points, dtype=np.int32)

                    # Draw smooth line through points
                    cv2.polylines(img, [pts], False, LASSO_COLOR, LASSO_THICKNESS, cv2.LINE_AA)

                    # Draw closing line if enough points
                    if len(state.lasso_points) > 2:
                        # Calculate distance between first and last point
                        close_distance = point_distance(state.lasso_points[0], state.lasso_points[-1])

                        # Draw closing line with distance-based alpha
                        alpha = min(1.0, close_distance / MIN_POINT_DISTANCE)
                        closing_color = tuple(int(c * alpha) for c in LASSO_COLOR)
                        cv2.line(img,
                                 state.lasso_points[-1],
                                 state.lasso_points[0],
                                 closing_color, LASSO_THICKNESS, cv2.LINE_AA)

            elif gesture == "FiveFingers" and state.is_lasso_mode:
                if len(state.lasso_points) > 2:
                    # Smooth the final selection by adding closing point
                    state.lasso_points.append(state.lasso_points[0])

                    # Create smooth mask
                    mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                    points = np.array(state.lasso_points, dtype=np.int32)

                    # Draw filled polygon with anti-aliasing
                    cv2.fillPoly(mask, [points], 255)

                    # Apply stronger Gaussian blur to mask edges
                    mask = cv2.GaussianBlur(mask, (5, 5), 2)
                    state.lasso_mask = mask

                    # Store complete background
                    mask_inv = cv2.bitwise_not(mask)
                    state.selection_background = cv2.bitwise_and(
                        state.original_canvas,
                        state.original_canvas,
                        mask=mask_inv
                    )

                    # Extract selection with smooth edges
                    state.lasso_selection = cv2.bitwise_and(
                        state.original_canvas,
                        state.original_canvas,
                        mask=mask
                    )
                    
                    # Extract selection from imgMarkerCanvas
                    state.lasso_selection_marker = cv2.bitwise_and(
                        state.original_marker_canvas,
                        state.original_marker_canvas,
                        mask=mask
                    )
                    # Store background from imgMarkerCanvas
                    state.selection_background_marker = cv2.bitwise_and(
                        state.original_marker_canvas,
                        state.original_marker_canvas,
                        mask=mask_inv
                    )                    
                    state.has_active_selection = True

                    # Initialize transform state
                    center_x = lmlist[9][1]
                    center_y = lmlist[9][2]
                    state.initial_lasso_pos = np.array([center_x, center_y])
                    state.initial_distance = calculate_hand_span(lmlist)

                state.is_lasso_mode = False

            elif gesture == "FiveFingers" and state.has_active_selection:
                center_x = lmlist[9][1]
                center_y = lmlist[9][2]
                current_pos = np.array([center_x, center_y])

                if state.initial_lasso_pos is not None:
                    # Calculate movement with smoothing
                    delta_pos = current_pos - state.initial_lasso_pos
                    if state.last_delta is not None:
                        delta_pos = smooth_transform(state.last_delta, delta_pos, TRANSFORM_SMOOTHING)
                    state.last_delta = delta_pos

                    # Calculate scaling with smoothing
                    current_distance = calculate_hand_span(lmlist)
                    if state.lasso_initial_distance != 0:
                        scale = current_distance / state.lasso_initial_distance
                        scale = smooth_transform(state.last_scale, scale, TRANSFORM_SMOOTHING)
                        state.last_scale = scale
                    else:
                        scale = 1.0

                    # Build the affine transformation matrix
                    cx, cy = state.initial_lasso_pos  # Use initial position as center
                    dx, dy = delta_pos[0], delta_pos[1]
                    s = scale

                    M = np.float32([
                        [s, 0, (1 - s) * cx + dx],
                        [0, s, (1 - s) * cy + dy]
                    ])

                    # Apply transformations with separate handling for marker and regular canvas
                    transformed = cv2.warpAffine(state.lasso_selection, M, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    transformed_marker = cv2.warpAffine(
                        state.lasso_selection_marker, M, (WINDOW_WIDTH, WINDOW_HEIGHT),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                    )
                    transformed_mask = cv2.warpAffine(state.lasso_mask, M, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    transformed_mask = cv2.GaussianBlur(transformed_mask, MASK_BLUR_SIZE, MASK_BLUR_SIGMA)

                    # Create inverted mask
                    mask_inv = cv2.bitwise_not(transformed_mask)

                    # Handle regular canvas
                    temp_canvas = state.selection_background.copy()
                    cv2.add(temp_canvas, transformed, temp_canvas, mask=transformed_mask)

                    # Handle marker canvas - preserve existing marker content
                    temp_marker_canvas = state.selection_background_marker.copy()
                    cv2.add(temp_marker_canvas, transformed_marker, temp_marker_canvas, mask=transformed_mask)

                    # Update both canvases atomically
                    imgCanvas[:] = temp_canvas
                    imgMarkerCanvas[:] = temp_marker_canvas

            elif not gesture == "Lasso" and state.has_active_selection:
                # Reset selection state when switching to other gestures
                imgCanvas, imgMarkerCanvas = reset_selection_state(state, imgCanvas, imgMarkerCanvas)
            else:
                state.drawing = False
                state.xp, state.yp = 0, 0
                state.first_frame = True

        else:
            state.drawing = False
            state.xp, state.yp = 0, 0
            state.first_frame = True

    # Combine the canvas with the live feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    # Now blend imgMarkerCanvas onto img
    img = cv2.addWeighted(img, 1, imgMarkerCanvas, 0.3, 0)
    
    # Always draw the header at the top of the image
    if header is not None:
        header_resized = cv2.resize(header, (WINDOW_WIDTH, HEADER_HEIGHT))
        img[0:HEADER_HEIGHT, 0:WINDOW_WIDTH] = header_resized

    # Display user feedback (mode, tool, and selected color)
    mode_text = f"Mode: {state.current_mode} | Tool: {state.current_tool}"
    if state.is_paused:
        mode_text += " | Paused"
    cv2.putText(img, mode_text, (10, WINDOW_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Create fixed-size window
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow("Image", img)

    # Combine canvases for display in "Canvas" window
    combined_canvas = imgCanvas.copy()
    cv2.addWeighted(combined_canvas, 1, imgMarkerCanvas, 0.3, 0, combined_canvas)

    # Display combined canvas in a separate window
    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Canvas", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow("Canvas", combined_canvas)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):  # Add help key
        show_instructions()
    elif key == ord('s'):
        save_drawing()
    elif key == ord('r'):
        imgCanvas = rotate_image(imgCanvas, 90)
        imgMarkerCanvas = rotate_image(imgMarkerCanvas, 90)
        print("Canvas rotated by 90 degrees.")
    elif key == ord('c'):
        clear_canvas()
    elif key == ord('z'):
        if state.command_index >= 0:
            state.command_index -= 1
            redraw_canvas()
        else:
            print("Nothing to undo.")
    elif key == ord('y'):
        if state.command_index < len(state.commands) - 1:
            state.command_index += 1
            state.commands[state.command_index].execute(imgCanvas, imgMarkerCanvas)
            redraw_canvas()
        else:
            print("Nothing to redo.")

# Clean up
cap.release()
cv2.destroyAllWindows()