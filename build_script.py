import PyInstaller.__main__
import sys
import os
import tempfile
import shutil
import logging
import mediapipe as mp
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_info_plist():
    content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>AirCanvas needs access to your camera for hand tracking.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>AirCanvas needs access to your microphone.</string>
    <key>NSCameraUseContinuityCameraDeviceType</key>
    <true/>
    <key>CFBundleExecutable</key>
    <string>AirCanvas</string>
    <key>CFBundleIdentifier</key>
    <string>com.aircanvas.app</string>
    <key>CFBundleName</key>
    <string>AirCanvas</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>NSCameraDevice</key>
    <true/>
    <key>NSCameraAccess</key>
    <true/>
    <key>NSCameraDefaultUsageDescription</key>
    <string>AirCanvas requires camera access for hand tracking functionality</string>
</dict>
</plist>'''
    return content

def verify_resources(base_path):
    header_path = os.path.join(base_path, 'Header')
    instructions_path = os.path.join(base_path, 'Instructions')
    
    logger.info(f"Checking resources in: {base_path}")
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"Header directory not found at {header_path}")
    if not os.path.exists(instructions_path):
        raise FileNotFoundError(f"Instructions directory not found at {instructions_path}")
    
    header_files = os.listdir(header_path)
    instructions_files = os.listdir(instructions_path)
    
    logger.info(f"Found Header files: {header_files}")
    logger.info(f"Found Instructions files: {instructions_files}")
    
    return True

def update_instructions_script(instructions_script):
    with open(instructions_script, 'r') as f:
        content = f.read()
    
    resource_handler = '''
import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
'''
    
    # Add resource path handler to start of file
    content = resource_handler + content
    
    # Update image loading paths
    content = content.replace(
        'os.path.join("Instructions",',
        'os.path.join(get_resource_path("Instructions"),'
    )
    
    with open(instructions_script, 'w') as f:
        f.write(content)
    logger.info("Updated instructions script resource paths")
    
def create_dmg_script():
    content = '''#!/bin/bash
    
# Set source directory
SOURCE_DIR="dist"
APP_NAME="AirCanvas.app"
DMG_NAME="AirCanvas-1.0.dmg"
VOLUME_NAME="AirCanvas"

# Create temporary DMG
TMP_DMG="tmp.dmg"
VOLUME_PATH="/Volumes/$VOLUME_NAME"

# Create DMG
hdiutil create -size 200m -fs HFS+ -volname "$VOLUME_NAME" "$TMP_DMG"

# Mount DMG
hdiutil attach "$TMP_DMG"

# Copy app
cp -R "$SOURCE_DIR/$APP_NAME" "$VOLUME_PATH"

# Create Applications symlink
ln -s /Applications "$VOLUME_PATH/Applications"

# Set permissions
chmod -Rf go-w "$VOLUME_PATH"

# Unmount
hdiutil detach "$VOLUME_PATH"

# Convert to compressed DMG
hdiutil convert "$TMP_DMG" -format UDZO -o "$DMG_NAME"

# Cleanup
rm "$TMP_DMG"
'''
    with open('create_dmg.sh', 'w') as f:
        f.write(content)
    os.chmod('create_dmg.sh', 0o755)
def check_camera_permissions():
    try:
        result = subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to get the value of (get the value of (every process whose name is "AirCanvas"))'],
            capture_output=True,
            text=True
        )
        return "AirCanvas" in result.stdout
    except Exception as e:
        logger.error(f"Error checking camera permissions: {e}")
        return False

def build():
    base_path = os.path.abspath(os.path.dirname(__file__))
    script_path = os.path.join(base_path, 'air_canvas_hand.py')
    instructions_script = os.path.join(base_path, 'air_canvas_instructions.py')
    
    # Verify resources exist
    verify_resources(base_path)
    
    # Update scripts with resource path handling
    update_instructions_script(instructions_script)
    
    # Get MediaPipe path
    mp_path = os.path.dirname(mp.__file__)
    
    # Create data args based on platform
    if sys.platform == 'darwin':
        if not check_camera_permissions():
            logger.error("Camera permissions not granted")
            raise RuntimeError("Please grant camera permissions in System Preferences")
        data_args = [
            f'--add-data={os.path.join(base_path, "Header")}:Header',
            f'--add-data={os.path.join(base_path, "Instructions")}:Instructions',
            f'--add-data={instructions_script}:.',
            f'--add-data={mp_path}:mediapipe'
        ]
    else:
        data_args = [
            f'--add-data={os.path.join(base_path, "Header")};Header',
            f'--add-data={os.path.join(base_path, "Instructions")};Instructions',
            f'--add-data={instructions_script};.',
            f'--add-data={mp_path};mediapipe'
        ]

    # PyInstaller arguments
    args = [
        script_path,
        '--name=AirCanvas',
        '--onefile',
        '--windowed',
        '--clean',
        *data_args,
        '--collect-all=mediapipe',
        '--hidden-import=mediapipe',
        '--hidden-import=cv2',
        '--hidden-import=tensorflow',
        '--hidden-import=PyQt5'
    ]

    try:
        # Run PyInstaller
        logger.info("Starting PyInstaller build...")
        PyInstaller.__main__.run(args)
        logger.info("PyInstaller build completed")
        
        # Post-process app bundle
        app_path = os.path.join(base_path, 'dist', 'AirCanvas.app')
        if os.path.exists(app_path):
            # Create MacOS directory
            macos_path = os.path.join(app_path, 'Contents', 'MacOS')
            os.makedirs(macos_path, exist_ok=True)
            
            # Copy resources to MacOS directory
            for folder in ['Header', 'Instructions']:
                src = os.path.join(base_path, folder)
                dst = os.path.join(macos_path, folder)
                if os.path.exists(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.info(f"Copied {folder} to {dst}")
                    # Verify contents
                    files = os.listdir(dst)
                    logger.info(f"{folder} contents at {dst}: {files}")
            
            # Write Info.plist
            plist_path = os.path.join(app_path, 'Contents', 'Info.plist')
            with open(plist_path, 'w') as f:
                f.write(create_info_plist())
            logger.info(f"Created Info.plist at {plist_path}")
            
            # Set executable permissions
            executable_path = os.path.join(macos_path, 'AirCanvas')
            if os.path.exists(executable_path):
                os.chmod(executable_path, 0o755)
                logger.info(f"Set executable permissions on {executable_path}")
            
            logger.info("Build completed successfully")
            
        # Create DMG script
        create_dmg_script()
        
        # Create DMG
        os.system('./create_dmg.sh')
        logger.info("DMG created successfully")
            
    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise

if __name__ == '__main__':
    build()