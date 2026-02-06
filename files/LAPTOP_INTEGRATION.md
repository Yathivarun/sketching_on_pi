# 💻 Laptop Integration Guide

This guide explains how to integrate the Pi sensor with your existing Sketch AI system on the laptop.

---

## 📋 What's New

The existing `app.py` already has Pi integration built-in! Here's what it does:

### Automatic Features

1. **Network Server**: Listens on port 5000 for Pi connections
2. **Embedding Matching**: Receives embeddings from Pi and matches against local DB
3. **Image Transmission**: Sends matched sketch images back to Pi
4. **Monitoring Page**: `/pi_status` shows Pi connection status

### No Changes Needed

✅ Your existing app.py already includes:
- `LaptopServer` from `network_protocol.py`
- Embedding handler: `handle_pi_embedding()`
- Status endpoint: `/api/pi/status`
- Pi monitoring page: `pi_sensor.html`

---

## 🚀 Running the System

### Start Order

1. **Laptop First**
   ```bash
   cd /path/to/sketch_ai
   python3 app.py
   ```
   
   Wait for:
   ```
   [PI] Starting Pi network server...
   [PI] ✓ Server started on port 5000
   [PI] Waiting for Pi to connect...
   ```

2. **Then Start Pi**
   ```bash
   # On Raspberry Pi
   cd ~/sketch_ai_pi
   python3 pi_main.py
   ```
   
   Laptop should show:
   ```
   [PI] ✓ Pi connected from ('192.168.137.198', XXXXX)
   ```

---

## 🌐 Network Configuration

### Laptop Side

**Option 1: Ethernet Internet Sharing (Recommended)**

**Windows:**
1. Settings → Network & Internet → Mobile hotspot
2. Share over: Ethernet
3. Or: Network Connections → Right-click WiFi → Properties → Sharing
4. Select Ethernet adapter

**macOS:**
1. System Preferences → Sharing
2. Internet Sharing: Share from WiFi to Ethernet
3. Turn on Internet Sharing

**Linux:**
```bash
# Using NetworkManager
sudo nmcli connection modify "Wired connection 1" \
    ipv4.method shared \
    ipv4.addresses 192.168.137.1/24

sudo nmcli connection up "Wired connection 1"
```

**Option 2: Static IP**

Set your Ethernet adapter to:
- IP: `192.168.137.1`
- Subnet: `255.255.255.0`
- Gateway: (leave empty)

---

## 📊 Monitoring Pi Status

### Web Interface

Visit: **http://localhost:8000/pi_status**

Shows:
- ✅ Connection status (Connected/Disconnected)
- 📊 Statistics
  - Embeddings received
  - Matches sent
  - Uptime
- 📝 Live activity feed
- ⚙️ Network configuration

### Console Logs

Watch laptop terminal for:
```
[PI] Received embedding at 2026-02-06 14:23:15
     Shape: (512,), Sample: [0.123 -0.456 0.789]
[PI] MATCH 1001 0.876
     → Sent 3 images to Pi
```

---

## 🔧 Troubleshooting

### Pi Not Connecting

**Check 1: Firewall**
```bash
# Windows: Allow Python through firewall
# macOS/Linux: Check firewall rules

# Test port is open
netstat -an | grep 5000
# Should show: LISTENING
```

**Check 2: Network**
```bash
# Ping Pi from laptop
ping 192.168.137.198

# Should get responses
```

**Check 3: Server Status**

Laptop should show:
```
[PI] Starting Pi network server...
[PI] ✓ Server started on port 5000
```

If not, check `network_protocol.py` is present.

### Embeddings Not Matching

**Check 1: Database**

Ensure you have preprocessed people:
```bash
ls preprocessed_data/
# Should show person IDs: 1001, 1002, etc.

ls preprocessed_data/1001/
# Should show: face_embedding.npy
```

**Check 2: Threshold**

In `recognition_manager.py`:
```python
self.similarity_threshold = 0.50  # Lower = more lenient
```

**Check 3: Embedding Format**

Laptop logs should show:
```
[PI] Received embedding at ...
     Shape: (512,), Sample: [...]
```

If shape is wrong, check Pi model loading.

### Images Not Sent to Pi

**Check 1: Sketch Files Exist**
```bash
ls outputs/generated_sketches/1001/
# Should show: *_scene_*.jpg files
```

If missing, run generation:
```bash
# Via web interface
curl -X POST http://localhost:8000/generate/1001
```

**Check 2: Laptop Logs**

Should show:
```
[LAPTOP-SERVER] Sent 3 images to Pi
```

If not, check `handle_pi_embedding()` function.

---

## 📁 File Structure

```
sketch_ai/                    (Laptop)
├── app.py                    ← Already has Pi integration
├── network_protocol.py       ← Shared between laptop & Pi
├── preprocess.py
├── generate.py
├── recognition_manager.py
├── camera_manager.py
├── templates/
│   └── pi_sensor.html        ← Pi monitoring page
├── preprocessed_data/
│   ├── 1001/
│   │   └── face_embedding.npy
│   └── 1002/
│       └── face_embedding.npy
└── outputs/
    └── generated_sketches/
        ├── 1001/
        │   ├── 1001_scene_1.jpg
        │   ├── 1001_scene_2.jpg
        │   └── 1001_scene_3.jpg
        └── 1002/
            └── ...
```

---

## 🔄 Workflow

### Full Cycle

1. **Pi captures face** (user presses SPACE)
2. **Pi detects face** using SCRFD
3. **Pi generates embedding** using ArcFace (512D)
4. **Pi sends embedding** to laptop via TCP
5. **Laptop receives embedding**
6. **Laptop matches** against `recognizer.find_match()`
7. **If HIT:**
   - Laptop loads sketch images from `outputs/generated_sketches/{person_id}/`
   - Laptop sends images to Pi
   - **Both laptop AND Pi display** the matched person
8. **If MISS:**
   - Laptop sends "no match" result
   - Pi continues slideshow

### Simultaneous Display

When a match occurs:
- **Laptop Display** (`/display`): Shows matched sketches
- **Pi Display**: Shows same matched sketches
- Both cycle through images for configured duration
- Both return to idle state (laptop: queue, Pi: slideshow)

---

## ⚙️ Configuration

### Adjust Pi Behavior

Edit `app.py` (laptop side):

```python
# In handle_pi_embedding()

# Change match threshold
recognizer.similarity_threshold = 0.50  # Default

# Change which images to send
scenes = sorted(person_dir.glob("*_scene_*.jpg"))  # All scenes
# Or:
scenes = sorted(person_dir.glob("*_scene_1.jpg"))  # Only first scene
```

### Adjust Display Duration

Edit `pi_config.py` (Pi side):

```python
MATCH_DISPLAY_INTERVAL = 5.0  # Seconds per image
MATCH_DISPLAY_CYCLES = 2       # How many full cycles
```

---

## 🧪 Testing

### Test Connection

**On laptop:**
```bash
python3 network_protocol.py laptop
```

**On Pi:**
```bash
python3 network_protocol.py pi
```

Should see connection established.

### Test Embedding Transmission

**On Pi:**
```python
# In pi_main.py or test script
import numpy as np
from network_protocol import PiClient

client = PiClient()
if client.connect():
    # Send random embedding
    test_emb = np.random.rand(512).astype(np.float32)
    client.send_embedding(test_emb)
    
    # Wait for response
    import time
    time.sleep(5)
```

**On laptop:**
Check terminal for:
```
[PI] Received embedding at ...
```

---

## 📈 Performance

### Expected Times

**Pi Side:**
- Face detection: 0.5-2 seconds (Pi 4)
- Embedding generation: 0.3-1 second
- Network transmission: <0.1 seconds

**Laptop Side:**
- Receive embedding: instant
- Match against DB: <0.01 seconds (for <100 people)
- Load images: <0.1 seconds
- Send images: 0.1-0.5 seconds (depends on image size)

**Total:** ~2-4 seconds from capture to display

### Optimization Tips

1. **Reduce Image Size**: Lower resolution sketches = faster transfer
2. **Limit Scenes**: Only send 1-2 scene images instead of all
3. **Cache Models**: Already implemented (models load once)
4. **Use Wired Connection**: Ethernet faster than WiFi

---

## 🔒 Security Notes

### Network Isolation

The Pi-Laptop connection is:
- ✅ Offline (no internet required)
- ✅ Direct Ethernet (not exposed to network)
- ✅ Custom protocol (not HTTP)

### Data Privacy

- Embeddings are 512D vectors (not raw images)
- Images only sent when matched
- No data stored permanently on Pi (except slideshow images)

---

## 📝 Logs

### Laptop Logs

```
[PI] Starting Pi network server...
[PI] ✓ Server started on port 5000
[PI] Waiting for Pi to connect...
[PI] ✓ Pi connected from ('192.168.137.198', 54321)
[PI] Received embedding at 2026-02-06 14:23:15
     Shape: (512,), Sample: [0.123 -0.456 0.789]
[PI] MATCH 1001 0.876
     → Sent 3 images to Pi
```

### Pi Logs

```
[NETWORK] Connecting to laptop at 192.168.137.1:5000...
[PI-CLIENT] ✓ Connected to laptop!
[CAPTURE] Processing frame...
[CAPTURE] ✓ Detected 1 face(s) in 1.23s
[CAPTURE] Face confidence: 0.987
[CAPTURE] ✓ Generated embedding in 0.45s
[CAPTURE] ✓ Sent to laptop
[MATCH] ✓ HIT: Person #1001 (confidence: 87.6%)
[IMAGES] ✓ Received 3 images from laptop
[DISPLAY] 🎯 MATCH TRIGGERED: Person #1001 (score: 0.876)
```

---

## ✅ Checklist

Before running, ensure:

- [ ] Laptop server is running (`app.py`)
- [ ] Port 5000 is listening
- [ ] Network configured (192.168.137.1 ↔ 192.168.137.198)
- [ ] Ethernet cable connected
- [ ] Database has preprocessed people
- [ ] Sketch images exist in `outputs/generated_sketches/`
- [ ] Pi models are installed
- [ ] Pi stock images added

---

## 🆘 Support

If issues persist:

1. **Check both terminals** for error messages
2. **Test network** with ping
3. **Verify models** are loaded correctly
4. **Check file paths** match configuration
5. **Review logs** for specific errors

---

**Integration Complete!** 🎉

Your existing system now seamlessly works with the Pi sensor!
