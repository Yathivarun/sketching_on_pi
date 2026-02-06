# 🔧 Raspberry Pi Troubleshooting Guide

## 🎥 Camera Issues

### Issue: "Failed to read frame" / Camera shuts down immediately

**Symptoms:**
```
[CAMERA] Opening camera 0...
[CAMERA] Warming up camera...
[ERROR] Failed to read frame
[SYSTEM] Shutdown initiated...
```

**Solutions:**

#### Step 1: Test Camera First
Before running the main app, test your camera:

```bash
cd ~/sketch_ai_pi
python3 test_camera.py
```

This will:
- Check for video devices
- Test frame capture
- Show live preview
- Diagnose common issues

#### Step 2: Check Camera Hardware

**For USB Webcam:**
```bash
# List video devices
ls -l /dev/video*

# Should show: /dev/video0, /dev/video1, etc.

# Test with v4l2 tools
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

**For Pi Camera Module:**
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable → Reboot

# Test with raspistill
raspistill -o test.jpg

# If this works but OpenCV doesn't, install:
sudo apt install python3-picamera2
```

#### Step 3: Try Different Camera IDs

Edit `pi_config.py`:
```python
CAMERA_ID = 1  # Try 1, 2, 3 if 0 fails
```

Or test directly:
```bash
python3 test_camera.py --camera 1
```

#### Step 4: Change Camera Backend

Edit `pi_config.py`:
```python
import cv2
CAMERA_BACKEND = cv2.CAP_V4L2  # For Linux V4L2
# Or
CAMERA_BACKEND = cv2.CAP_GSTREAMER  # For GStreamer
```

#### Step 5: Reduce Resolution

If camera works but is slow/unstable, reduce resolution in `pi_config.py`:
```python
CAMERA_WIDTH = 320   # Instead of 640
CAMERA_HEIGHT = 240  # Instead of 480
CAMERA_WARMUP_FRAMES = 50  # Increase warmup
```

#### Step 6: Check Permissions

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again for changes to take effect
# Or reboot
sudo reboot
```

#### Step 7: Reinstall OpenCV

If camera still fails:
```bash
# Uninstall
pip3 uninstall opencv-python opencv-python-headless

# Reinstall with proper build
pip3 install opencv-python==4.8.1.78

# If issues persist, try system package
sudo apt install python3-opencv
```

---

## 🌐 Network Issues

### Issue: "Connection failed" / Cannot reach laptop

**Symptoms:**
```
[NETWORK] Connecting to laptop at 192.168.137.1:5000...
[NETWORK] ✗ Connection failed
```

**Solutions:**

#### Step 1: Test Network Connectivity

```bash
# Ping laptop from Pi
ping 192.168.137.1

# Should see replies
# If "Destination Host Unreachable" → network not configured
# If "100% packet loss" → firewall or wrong IP
```

#### Step 2: Verify IP Configuration

**On Pi:**
```bash
# Check Pi's IP
ip addr show eth0
# Should show: inet 192.168.137.198/24

# If wrong, fix in /etc/dhcpcd.conf
sudo nano /etc/dhcpcd.conf

# Add at end:
interface eth0
static ip_address=192.168.137.198/24
static routers=192.168.137.1

# Save and reboot
sudo reboot
```

**On Laptop:**
Check laptop has IP 192.168.137.1 on Ethernet adapter

#### Step 3: Check Laptop Server is Running

**On Laptop:**
```bash
# Start the app
python3 app.py

# Should see:
# [PI] ✓ Server started on port 5000
# [PI] Waiting for Pi to connect...

# Verify port is listening
netstat -an | grep 5000
# Should show: LISTEN on port 5000
```

#### Step 4: Check Firewall

**On Laptop (Linux):**
```bash
# Allow port 5000
sudo ufw allow 5000/tcp

# Or disable firewall temporarily for testing
sudo ufw disable
```

**On Laptop (Windows):**
- Windows Defender Firewall
- Allow Python through firewall
- Or: Advanced Settings → Inbound Rules → New Rule → Port 5000 TCP

**On Laptop (macOS):**
- System Preferences → Security & Privacy → Firewall
- Firewall Options → Add Python → Allow

#### Step 5: Test with Different Port

Edit `pi_config.py` and laptop's `network_protocol.py`:
```python
LAPTOP_PORT = 5001  # Try different port
```

---

## 🤖 Model Issues

### Issue: "Model not found"

**Symptoms:**
```
FileNotFoundError: Model not found: ~/.insightface/models/buffalo_l/det_500m.onnx
```

**Solutions:**

#### Method 1: Automatic Download
```bash
pip3 install insightface

python3 << EOF
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # -1 for CPU
print("Models downloaded!")
EOF

pip3 uninstall insightface  # Optional: remove after download
```

#### Method 2: Manual Download
```bash
cd ~/.insightface/models/buffalo_l

# Download detector
wget https://github.com/deepinsight/insightface/releases/download/v0.7/det_500m.onnx

# Download recognizer  
wget https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx

# Verify
ls -lh
# Should show both .onnx files
```

#### Method 3: Copy from Laptop

If models already on laptop:
```bash
# On laptop
cd ~/.insightface/models/buffalo_l
tar czf models.tar.gz *.onnx

# Transfer to Pi (via USB or network)
scp models.tar.gz pi@192.168.137.198:~/

# On Pi
mkdir -p ~/.insightface/models/buffalo_l
tar xzf ~/models.tar.gz -C ~/.insightface/models/buffalo_l/
```

---

## ⚡ Performance Issues

### Issue: Very slow inference (>5 seconds per frame)

**Solutions:**

#### 1. Reduce Camera Resolution
```python
# pi_config.py
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
```

#### 2. Optimize ONNX Threads
```python
# pi_config.py
ONNX_INTRA_THREADS = 4  # Match Pi CPU cores
ONNX_INTER_THREADS = 1
```

#### 3. Check CPU Frequency
```bash
# Check current frequency
vcgencmd measure_clock arm

# If throttled, check temperature
vcgencmd measure_temp

# If >80°C, improve cooling
```

#### 4. Overclock (Advanced)
```bash
sudo nano /boot/config.txt

# Add (for Pi 4):
over_voltage=6
arm_freq=2000

# Reboot
sudo reboot
```

⚠️ **Warning:** Overclocking can void warranty and cause instability

#### 5. Close Other Applications
```bash
# Check CPU usage
top

# Kill unnecessary processes
```

---

## 📦 Dependency Issues

### Issue: Import errors / Module not found

**Symptoms:**
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Solutions:**

```bash
# Reinstall requirements
pip3 install -r requirements_pi.txt --force-reinstall

# If specific package fails, install individually
pip3 install opencv-python-headless==4.8.1.78
pip3 install numpy==1.24.3
pip3 install onnxruntime==1.16.3

# Check installations
python3 -c "import cv2; import numpy; import onnxruntime; print('All OK')"
```

---

## 🖼️ Display Issues

### Issue: No stock images / Black screen

**Symptoms:**
Display shows black or "Waiting for face detection"

**Solutions:**

```bash
# Check stock images exist
ls ~/sketch_ai_pi/stock_images/

# If empty, add images
# Supported: .jpg, .jpeg, .png, .bmp

# Download sample images
cd ~/sketch_ai_pi/stock_images
wget https://picsum.photos/800/600 -O sample1.jpg
wget https://picsum.photos/800/600 -O sample2.jpg
wget https://picsum.photos/800/600 -O sample3.jpg

# Verify
ls -lh
```

### Issue: Matched images not displaying

**Solutions:**

1. **Check laptop sent images:**
   ```
   # Laptop logs should show:
   [LAPTOP-SERVER] Sent 3 images to Pi
   ```

2. **Check Pi received images:**
   ```bash
   ls ~/sketch_ai_pi/received_images/
   # Should show .jpg files
   ```

3. **Check display mode:**
   Pi logs should show:
   ```
   [DISPLAY] 🎯 MATCH TRIGGERED: Person #1001
   ```

---

## 🔌 Connection Drops / Disconnects

### Issue: Pi keeps disconnecting

**Symptoms:**
```
[NETWORK] ✗ Disconnected from laptop
[NETWORK] Attempting to reconnect...
```

**Solutions:**

#### 1. Check Ethernet Cable
- Try different cable
- Check both ends are secure
- Look for physical damage

#### 2. Increase Heartbeat Timeout
```python
# network_protocol.py
CONNECTION_TIMEOUT = 30.0  # Increase from 15.0
HEARTBEAT_INTERVAL = 10.0  # Increase from 5.0
```

#### 3. Disable Power Management
```bash
# Disable Ethernet power management
sudo nano /etc/network/interfaces

# Add:
auto eth0
iface eth0 inet static
    address 192.168.137.198
    netmask 255.255.255.0
    gateway 192.168.137.1
    post-up /sbin/ethtool -s eth0 wol d
    post-up /sbin/ethtool --set-eee eth0 eee off
```

---

## 🐛 General Debugging

### Enable Verbose Logging

Edit Python files to add more debug output:

```python
# At top of pi_main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Resources

```bash
# CPU and Memory
htop

# Temperature
vcgencmd measure_temp

# Disk space
df -h

# System info
cat /proc/cpuinfo | grep Model
```

### Test Individual Components

```bash
# Test camera only
python3 test_camera.py

# Test network only
python3 network_protocol.py pi

# Test display only
python3 pi_display.py
```

---

## 📋 Quick Diagnostic Checklist

Run through this checklist when troubleshooting:

```bash
# 1. System Requirements
python3 --version  # Should be 3.7+
uname -m  # Should be aarch64 or armv7l

# 2. Dependencies
python3 -c "import cv2; import numpy; import onnxruntime; print('OK')"

# 3. Camera
ls /dev/video*
python3 test_camera.py

# 4. Models
ls ~/.insightface/models/buffalo_l/

# 5. Network
ping 192.168.137.1
nc -zv 192.168.137.1 5000

# 6. Stock Images
ls ~/sketch_ai_pi/stock_images/ | wc -l  # Should be > 0

# 7. Permissions
groups $USER | grep video  # Should include 'video'
```

---

## 🆘 Still Having Issues?

### Collect Debug Information

```bash
# Create debug report
cat > debug_report.txt << EOF
System Info:
$(cat /proc/cpuinfo | grep Model)
$(uname -a)
$(python3 --version)

Python Packages:
$(pip3 list | grep -E "opencv|numpy|onnx")

Camera Devices:
$(ls -l /dev/video*)

Network Config:
$(ip addr show eth0)
$(ping -c 3 192.168.137.1 2>&1 | tail -5)

Models:
$(ls -lh ~/.insightface/models/buffalo_l/ 2>&1)

Stock Images:
$(ls ~/sketch_ai_pi/stock_images/ | wc -l)

Last Error:
$(tail -50 /tmp/pi_error.log 2>&1)
EOF

cat debug_report.txt
```

### Reset to Known Good State

```bash
# Backup current setup
cd ~/sketch_ai_pi
tar czf backup_$(date +%Y%m%d).tar.gz *.py stock_images/

# Fresh install
cd ~
rm -rf sketch_ai_pi
mkdir sketch_ai_pi
cd sketch_ai_pi

# Copy fresh files from laptop
# Follow PI_SETUP_README.md from scratch
```

---

## 📞 Common Error Messages Reference

| Error | Likely Cause | Fix Section |
|-------|--------------|-------------|
| "Failed to read frame" | Camera hardware | Camera Issues |
| "Connection failed" | Network config | Network Issues |
| "Model not found" | Models not installed | Model Issues |
| "No face detected" (every time) | Poor lighting or camera angle | Camera Issues |
| "Slow performance" | Pi overloaded | Performance Issues |
| "Module not found" | Missing dependencies | Dependency Issues |
| Black display | No stock images | Display Issues |
| Keeps disconnecting | Network unstable | Connection Drops |

---

**Last Updated:** February 6, 2026
