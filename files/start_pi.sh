#!/bin/bash
# Quick Start Script for Raspberry Pi Sensor
# Run this to check installation and start the system

set -e  # Exit on error

echo "=================================="
echo "🔷 Sketch AI Pi - Quick Start"
echo "=================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.7.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3.7+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Check if in correct directory
if [ ! -f "pi_main.py" ]; then
    echo -e "${RED}✗ Error: pi_main.py not found${NC}"
    echo "Please run this script from the sketch_ai_pi directory"
    exit 1
fi

# Check required files
echo ""
echo "Checking required files..."
FILES=("pi_main.py" "pi_face_detect.py" "pi_display.py" "pi_config.py" "network_protocol.py")
ALL_FOUND=true

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file (missing)${NC}"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo -e "${RED}Please copy all required files to this directory${NC}"
    exit 1
fi

# Check Python packages
echo ""
echo "Checking Python packages..."
PACKAGES=("cv2:opencv-python-headless" "numpy:numpy" "onnxruntime:onnxruntime")

for pkg in "${PACKAGES[@]}"; do
    IFS=':' read -r import_name package_name <<< "$pkg"
    if python3 -c "import $import_name" 2>/dev/null; then
        echo -e "${GREEN}✓ $package_name${NC}"
    else
        echo -e "${RED}✗ $package_name (not installed)${NC}"
        echo "  Install: pip3 install $package_name"
        exit 1
    fi
done

# Check InsightFace models
echo ""
echo "Checking InsightFace models..."
MODEL_DIR="$HOME/.insightface/models/buffalo_l"

if [ -f "$MODEL_DIR/det_500m.onnx" ]; then
    echo -e "${GREEN}✓ Face detector model${NC}"
else
    echo -e "${RED}✗ Face detector model (not found)${NC}"
    echo "  Expected: $MODEL_DIR/det_500m.onnx"
    echo "  See PI_SETUP_README.md for installation"
    exit 1
fi

if [ -f "$MODEL_DIR/w600k_r50.onnx" ]; then
    echo -e "${GREEN}✓ Face recognizer model${NC}"
else
    echo -e "${RED}✗ Face recognizer model (not found)${NC}"
    echo "  Expected: $MODEL_DIR/w600k_r50.onnx"
    echo "  See PI_SETUP_README.md for installation"
    exit 1
fi

# Check stock images
echo ""
echo "Checking stock images..."
if [ -d "stock_images" ]; then
    IMAGE_COUNT=$(find stock_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    if [ $IMAGE_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓ Found $IMAGE_COUNT stock images${NC}"
    else
        echo -e "${YELLOW}⚠ No stock images found${NC}"
        echo "  Add images to: $(pwd)/stock_images/"
        echo "  System will work but display will be empty during idle"
    fi
else
    echo -e "${YELLOW}⚠ stock_images directory not found${NC}"
    echo "Creating directory..."
    mkdir -p stock_images
    echo "  Add images to: $(pwd)/stock_images/"
fi

# Check camera
echo ""
echo "Checking camera..."
if [ -c "/dev/video0" ]; then
    echo -e "${GREEN}✓ Camera device found${NC}"
else
    echo -e "${YELLOW}⚠ Camera not detected at /dev/video0${NC}"
    echo "  Make sure camera is connected"
    echo "  For Pi Camera, enable it in raspi-config"
fi

# Test network connectivity
echo ""
echo "Checking network connectivity..."
LAPTOP_IP="192.168.137.1"

if ping -c 1 -W 2 $LAPTOP_IP > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Laptop reachable at $LAPTOP_IP${NC}"
else
    echo -e "${YELLOW}⚠ Cannot reach laptop at $LAPTOP_IP${NC}"
    echo "  Make sure:"
    echo "  - Ethernet cable is connected"
    echo "  - Pi IP is set to 192.168.137.198"
    echo "  - Laptop IP is set to 192.168.137.1"
    echo ""
    echo "  You can continue, but connection will fail"
fi

# Check if laptop server is running
echo ""
echo "Checking laptop server..."
if timeout 2 bash -c "echo > /dev/tcp/$LAPTOP_IP/5000" 2>/dev/null; then
    echo -e "${GREEN}✓ Laptop server is running on port 5000${NC}"
else
    echo -e "${YELLOW}⚠ Cannot connect to laptop server${NC}"
    echo "  Make sure laptop is running: python3 app.py"
    echo ""
    echo "  You can continue, but connection will fail at runtime"
fi

# Summary
echo ""
echo "=================================="
echo "📋 Pre-flight Check Complete"
echo "=================================="
echo ""

# Ask to continue
read -p "Start Pi sensor now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=================================="
    echo "🚀 Starting Pi Sensor..."
    echo "=================================="
    echo ""
    echo "Controls:"
    echo "  SPACE - Capture and process face"
    echo "  Q     - Quit"
    echo ""
    echo "Starting in 3 seconds..."
    sleep 3
    
    # Run the main application
    python3 pi_main.py
else
    echo ""
    echo "Setup complete. Run manually with:"
    echo "  python3 pi_main.py"
    echo ""
fi
