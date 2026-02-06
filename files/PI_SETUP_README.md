# 🔷 Raspberry Pi Sensor Setup Guide

This guide will help you set up the Raspberry Pi as a remote face detection sensor for the Sketch AI system.

---

## 📋 Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Installation](#installation)
4. [Network Configuration](#network-configuration)
5. [Running the System](#running-the-system)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Hardware Requirements

- **Raspberry Pi 4** (4GB RAM recommended, 2GB minimum)
- **Pi Camera Module** or USB webcam
- **Display** (HDMI monitor for showing results)
- **Ethernet Cable** (for connecting to laptop)
- **Power Supply** (Official Pi 4 power adapter recommended)
- **MicroSD Card** (32GB minimum, Class 10)

**Note:** Raspberry Pi 3B+ works but may be slower. Pi Zero is NOT recommended.

---

## 💾 Software Requirements

### Operating System
- **Raspberry Pi OS** (64-bit recommended)
- Updated to latest version: `sudo apt update && sudo apt upgrade`

### Python Version
- **Python 3.7+** (usually pre-installed)
- Check: `python3 --version`

---

## 📦 Installation

### Step 1: System Dependencies

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install OpenCV dependencies
sudo apt install -y python3-opencv
sudo apt install -y libopencv-dev

# Install build tools
sudo apt install -y build-essential cmake
sudo apt install -y libatlas-base-dev
sudo apt install -y gfortran

# Install Python development headers
sudo apt install -y python3-dev python3-pip

# Install system libraries for ONNX Runtime
sudo apt install -y libgomp1
```

### Step 2: Python Packages

```bash
# Upgrade pip
pip3 install --upgrade pip

# Install core packages
pip3 install opencv-python-headless==4.8.1.78
pip3 install numpy==1.24.3
pip3 install onnxruntime==1.16.3

# Note: Use opencv-python-headless to avoid Qt conflicts
# If you need GUI, use opencv-python instead
```

### Step 3: InsightFace Models

```bash
# Create models directory
mkdir -p ~/.insightface/models/buffalo_l

# Download models (two options)
```

**Option A: Automatic Download (Recommended)**
```bash
# Install insightface package temporarily
pip3 install insightface

# Run Python to auto-download models
python3 << EOF
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # -1 for CPU
print("Models downloaded successfully!")
EOF

# Uninstall insightface (we only needed it for downloading)
pip3 uninstall insightface -y
```

**Option B: Manual Download**
```bash
# Download from InsightFace model zoo
cd ~/.insightface/models/buffalo_l

# Download detector
wget https://github.com/deepinsight/insightface/releases/download/v0.7/det_500m.onnx

# Download recognizer
wget https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx
```

**Verify Models:**
```bash
ls ~/.insightface/models/buffalo_l/
# Should show: det_500m.onnx  w600k_r50.onnx
```

### Step 4: Pi Project Files

Copy all Pi files to the Raspberry Pi:

```bash
# Create project directory
mkdir -p ~/sketch_ai_pi
cd ~/sketch_ai_pi

# Copy files from laptop (using scp or USB drive)
# Files needed:
# - pi_main.py
# - pi_face_detect.py
# - pi_display.py
# - pi_config.py
# - network_protocol.py (from laptop)

# If copying from laptop via network:
# scp user@laptop:/path/to/files/*.py ~/sketch_ai_pi/
```

### Step 5: Stock Images

```bash
# Create stock images directory
mkdir -p ~/sketch_ai_pi/stock_images

# Add your slideshow images here
# Supported formats: JPG, PNG, BMP
# Copy from USB drive or download
```

---

## 🌐 Network Configuration

### Ethernet Connection

**Option 1: Direct Connection (Recommended)**

1. **On Laptop:**
   - Go to Network Settings
   - Share Internet Connection over Ethernet
   - Set static IP: `192.168.137.1`
   - Subnet mask: `255.255.255.0`

2. **On Raspberry Pi:**
   ```bash
   sudo nano /etc/dhcpcd.conf
   ```
   
   Add at the end:
   ```
   interface eth0
   static ip_address=192.168.137.198/24
   static routers=192.168.137.1
   static domain_name_servers=192.168.137.1 8.8.8.8
   ```
   
   Save and reboot:
   ```bash
   sudo reboot
   ```

3. **Verify Connection:**
   ```bash
   # On Pi, ping laptop
   ping 192.168.137.1
   
   # Should see responses
   ```

**Option 2: Local Network**

Both devices on same WiFi/router:

1. **Find Pi IP:**
   ```bash
   hostname -I
   ```

2. **Update `pi_config.py`:**
   ```python
   LAPTOP_IP = "192.168.1.XXX"  # Your laptop's actual IP
   ```

---

## 🚀 Running the System

### Start Laptop Server First

On laptop:
```bash
# Make sure app.py is running
python3 app.py

# Should see:
# [PI] Starting Pi network server...
# [PI] ✓ Server started on port 5000
# [PI] Waiting for Pi to connect...
```

### Start Pi Sensor

On Raspberry Pi:
```bash
cd ~/sketch_ai_pi

# Run main application
python3 pi_main.py
```

**Expected Output:**
```
================================================================
🔷 INITIALIZING PI SYSTEM
================================================================

[1/3] Starting Display Manager...
[DISPLAY] Loading stock images...
[DISPLAY] ✓ Loaded 25 stock images
[DISPLAY] ✓ Display started

[2/3] Starting Face Capture...
[DETECTOR] Loading SCRFD model from ~/.insightface/models/buffalo_l/det_500m.onnx...
[DETECTOR] ✓ Loaded. Input size: (640, 640)
[RECOGNIZER] Loading ArcFace model from ~/.insightface/models/buffalo_l/w600k_r50.onnx...
[RECOGNIZER] ✓ Loaded. Embedding size: 512D

[3/3] Connecting to Laptop...
[NETWORK] Connecting to laptop at 192.168.137.1:5000...
[NETWORK] ✓ Connected to laptop

================================================================
✓ PI SYSTEM READY
================================================================
Display: Running in slideshow mode
Camera: Press SPACE to capture, Q to quit
Network: Connected
================================================================
```

### Usage

1. **Slideshow Mode (Default)**
   - Display shows random images from `stock_images/` folder
   - Camera window shows live preview

2. **Capture Face**
   - Press **SPACE** in camera window
   - System detects face
   - Generates embedding
   - Sends to laptop

3. **Match Result**
   - If **HIT**: Display shows person's sketch images
   - If **MISS**: Display continues slideshow

4. **Quit**
   - Press **Q** in camera window
   - Or `Ctrl+C` in terminal

---

## 🔍 Troubleshooting

### Connection Issues

**Problem: "Connection failed"**
```bash
# Check laptop is reachable
ping 192.168.137.1

# Check port is open on laptop
# On laptop: netstat -an | grep 5000

# Verify IP configuration
ip addr show eth0
```

### Camera Issues

**Problem: "Failed to open camera"**
```bash
# Check camera is detected
ls /dev/video*

# Test with simple OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# If using Pi Camera module, enable it
sudo raspi-config
# Interface Options → Camera → Enable
```

### Model Issues

**Problem: "Model not found"**
```bash
# Verify models exist
ls -lh ~/.insightface/models/buffalo_l/

# Should show two .onnx files
# If missing, re-run model download step
```

### Performance Issues

**Problem: "Slow inference"**

The Pi is CPU-only and will be slower than laptop. Typical times:
- Face detection: 0.5-2 seconds
- Embedding generation: 0.3-1 second

**Optimizations already applied:**
- CPU-only ONNX providers
- Reduced camera resolution (640x480)
- Efficient threading
- Model caching

**If still too slow:**
1. Reduce camera resolution in `pi_config.py`:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

2. Close other applications
3. Overclock Pi (advanced users)

### Display Issues

**Problem: "No stock images displayed"**
```bash
# Check stock images exist
ls ~/sketch_ai_pi/stock_images/

# Add images if empty
# Supported: .jpg, .jpeg, .png, .bmp
```

**Problem: "Display not fullscreen"**

Edit `pi_config.py`:
```python
DISPLAY_FULLSCREEN = True
```

---

## 📊 System Monitoring

### Check System Resources

```bash
# CPU usage
top

# Memory usage
free -h

# Temperature (important for Pi)
vcgencmd measure_temp
```

### Network Status

```bash
# Check connection
# On laptop: Visit http://localhost:8000/pi_status

# Shows:
# - Connection status
# - Embeddings received
# - Matches sent
# - Uptime
```

---

## 🔄 Auto-Start on Boot (Optional)

Create systemd service:

```bash
sudo nano /etc/systemd/system/sketch-ai-pi.service
```

```ini
[Unit]
Description=Sketch AI Pi Sensor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/sketch_ai_pi
ExecStart=/usr/bin/python3 /home/pi/sketch_ai_pi/pi_main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sketch-ai-pi.service
sudo systemctl start sketch-ai-pi.service

# Check status
sudo systemctl status sketch-ai-pi.service
```

---

## 📝 Configuration Reference

Edit `pi_config.py` to customize:

```python
# Network
LAPTOP_IP = "192.168.137.1"  # Change if different
LAPTOP_PORT = 5000

# Camera
CAMERA_ID = 0  # Change if using USB camera
CAMERA_WIDTH = 640  # Lower for better performance
CAMERA_HEIGHT = 480

# Display
DISPLAY_FULLSCREEN = True
SLIDESHOW_INTERVAL = 3.0  # Seconds per image
MATCH_DISPLAY_INTERVAL = 5.0
MATCH_DISPLAY_CYCLES = 2  # How many times to cycle matched images

# Performance
ONNX_INTRA_THREADS = 2  # Reduce if Pi is slow
```

---

## 🆘 Getting Help

1. **Check logs:** Errors are printed to terminal
2. **Verify network:** Both devices must be connected
3. **Test models:** Ensure InsightFace models are installed
4. **Monitor laptop:** Check `/pi_status` page on laptop

---

## ✅ Quick Checklist

- [ ] Raspberry Pi 4 with 4GB RAM
- [ ] Raspberry Pi OS updated
- [ ] Python 3.7+ installed
- [ ] OpenCV installed
- [ ] ONNX Runtime installed
- [ ] InsightFace models downloaded
- [ ] Pi files copied
- [ ] Stock images added
- [ ] Network configured (Pi: 192.168.137.198, Laptop: 192.168.137.1)
- [ ] Laptop server running
- [ ] Pi connected via Ethernet

If all checked, you're ready to run `python3 pi_main.py`! 🚀

---

**Last Updated:** February 2026
