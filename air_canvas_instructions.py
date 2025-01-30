
import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# air_canvas_instructions.py

from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os

class InstructionsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.current_page = 0
        
        # Set a larger initial size
        self.setMinimumSize(1000, 800)
        
        # Define titles and load images (same as before)
        self.titles = [
            "Welcome to Air Canvas!",
            "One Finger - Drawing Mode", 
            "Two Fingers - Selection Mode",
            "Three Fingers - Tool Reset",
            "Four Fingers - Eraser Mode", 
            "Five Fingers - Transform Mode",
            "Lasso Selection",
            "Pinch Gesture - Pause"
        ]
        
        # Load images (same as before)
        self.images = []
        for i in range(1, 9):
            image_path = os.path.join(get_resource_path("Instructions"), f"{i}.jpg")
            if os.path.exists(image_path):
                self.images.append(QPixmap(image_path))
            else:
                print(f"Warning: Could not find {image_path}")
                self.images.append(QPixmap())
        
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Air Canvas Instructions")
        
        # Main layout with margins
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Content widget
        content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(20)
        content.setLayout(self.content_layout)
        
        # Create horizontal button layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.close_btn = QPushButton("Start Drawing!")
        
        for btn in [self.prev_btn, self.next_btn, self.close_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    min-width: 150px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
        
        self.close_btn.setStyleSheet(self.close_btn.styleSheet() + """
            QPushButton {
                background-color: #2196F3;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addWidget(self.close_btn)
        btn_layout.addStretch()
        
        self.prev_btn.clicked.connect(self.previous_page)
        self.next_btn.clicked.connect(self.next_page)
        self.close_btn.clicked.connect(self.accept)
        
        # Add widgets to main layout
        layout.addWidget(content)
        layout.addLayout(btn_layout)
        
        # Set dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel[title="true"] {
                font-size: 28px;
                font-weight: bold;
                color: #1a237e;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
            }
        """)
        
        self.setLayout(layout)
        self.update_content()

    def update_content(self):
        # Clear previous content
        for i in reversed(range(self.content_layout.count())): 
            self.content_layout.itemAt(i).widget().setParent(None)
        
        # Create and style title
        title = QLabel(self.titles[self.current_page])
        title.setProperty("title", "true")
        title.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(title)
        
        # Create image container with white background
        image_container = QWidget()
        image_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 20px;
        """)
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create and scale image
        image_label = QLabel()
        pixmap = self.images[self.current_page]
        
        # Calculate available space
        available_height = self.height() - 200  # Account for title and buttons
        available_width = self.width() - 80     # Account for margins
        
        # Scale image to fit available space
        scaled_pixmap = pixmap.scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        
        image_layout.addWidget(image_label)
        image_container.setLayout(image_layout)
        
        self.content_layout.addWidget(image_container)
        
        # Update button states
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < len(self.titles) - 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_content()  # Rescale image when dialog is resized

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_content()
            
    def next_page(self):
        if self.current_page < len(self.titles) - 1:
            self.current_page += 1
            self.update_content()

def show_instructions():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    dialog = InstructionsDialog()
    dialog.exec_()