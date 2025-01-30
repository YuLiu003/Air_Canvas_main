#!/bin/bash
    
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
