#!/bin/bash
# Pre-build script to copy splash screen files
echo "Copying splash screen files..."

# Create dist directory if it doesn't exist
mkdir -p ../dist

# Copy splash screen files to dist
cp src/splashscreen.html ../dist/
cp src/splash-styles.css ../dist/

echo "Splash screen files copied successfully!"
