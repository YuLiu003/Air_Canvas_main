import math
import cv2
import mediapipe as mp
import sys
import os


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplex,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.is_pinching = False
        self.pinch_threshold = 30  # Adjust this value to change pinch sensitivity
        self.is_left_hand = False  # Will be updated after processing each frame

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Determine handedness for the first detected hand if any
        self.is_left_hand = False
        if self.results.multi_handedness:
            # We assume we are dealing with the first detected hand for simplicity
            handedness_label = self.results.multi_handedness[0].classification[0].label
            # handedness_label will be either "Left" or "Right"
            self.is_left_hand = (handedness_label == "Left")

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xmin, xmax = min(xmin, cx), max(xmax, cx)
                    ymin, ymax = min(ymin, cy), max(ymax, cy)
                    self.lmList.append([id, cx, cy])
                if draw:
                    cv2.rectangle(
                        img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                        (0, 255, 0), 2
                    )
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) != 0:
            # For the thumb:
            # Original condition (for right hand):
            # if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:

            # Adjust logic if left hand is detected.
            # On a right hand, the thumb is typically on the left side (smaller x-value).
            # On a left hand, the thumb would be on the right side (larger x-value).
            if self.is_left_hand:
                # Left hand: invert the condition
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # Right hand: original logic
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # For other fingers (index, middle, ring, pinky), the logic is usually vertical (y-axis based).
            # Typically, whether it's a left or right hand doesn't affect this logic,
            # because "up" is determined by a landmark being above another landmark on the y-axis.
            # If your logic for other fingers is directional on x-axis, you may need to invert similarly.
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def detectPinch(self):
        """Detect pinch gesture between thumb and index finger"""
        if len(self.lmList) >= 9:  # Make sure we have both thumb and index finger landmarks
            thumb_tip = self.lmList[4]
            index_tip = self.lmList[8]
            
            # Calculate distance between thumb and index finger tips
            distance = math.sqrt(
                (thumb_tip[1] - index_tip[1])**2 + 
                (thumb_tip[2] - index_tip[2])**2
            )
            
            # Update pinch state
            self.is_pinching = distance < self.pinch_threshold
            return self.is_pinching
        return False

    def detectGesture(self):
        fingers = self.fingersUp()
        if len(fingers) != 5:
            return None

        # Check for pinch gesture first
        if self.detectPinch():
            return "Pinch"
            
        # Define gestures based on fingers
        if fingers == [0, 1, 1, 1, 1]:
            return "FourFingers"
        if fingers == [0, 1, 1, 1, 0]:
            return "ThreeFingers"
        if fingers == [0, 1, 1, 0, 0]:
            return "TwoFingers"
        if fingers == [0, 1, 0, 0, 0]:
            return "OneFinger"
        if fingers == [1, 1, 1, 1, 1]:
            return "FiveFingers"
        if fingers == [1, 0, 0, 0, 1]:
            return "Lasso"
        return None