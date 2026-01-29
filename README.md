# Doom Scrolling Alarm

A macOS app that monitors your eyes through the camera and plays an alarm when it detects your eyes have been closed for too long. Perfect for preventing doom scrolling-induced naps.

## Features

- Real-time eye tracking using MediaPipe Face Landmarker
- Eye Aspect Ratio (EAR) detection for accurate open/closed eye detection
- Configurable threshold for how long eyes can be closed before alarm triggers
- Alarm video with audio that plays on loop until eyes are open again
- Optional preview window showing detection status
- Standalone macOS app - no Python installation required for end users

## Requirements

### For Running the Built App
- macOS 11.0 or later
- Camera access

### For Development/Building
- Python 3.10+
- pip

## Installation (Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DoomScrollingAlarm.git
   cd DoomScrollingAlarm
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python main.py
   ```

## Building the App

To create a standalone macOS application:

1. Ensure you have the development environment set up (see above)

2. Run the build script:
   ```bash
   python build.py
   ```

3. The built app will be located at `dist/DoomScrollingAlarm.app`

## Running the App

### From Source
```bash
python main.py
```

### Built App
```bash
open dist/DoomScrollingAlarm.app
```

Or double-click `DoomScrollingAlarm.app` in Finder.

**Note:** On first launch, macOS will ask for camera permission. Grant access to allow eye tracking.

If macOS shows a security warning:
1. Right-click (or Control-click) the app
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears

## Configuration

Edit `config.json` to customize the app behavior:

```json
{
  "threshold": 0.25,
  "interval": 0.5,
  "show_preview": true,
  "ear_threshold": 0.2,
  "paths": {
    "models_dir": "models",
    "model_file": "face_landmarker.task",
    "alarm_video": "alarm/skeleton banging shield.mp4"
  },
  "urls": {
    "model_download": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
  }
}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `threshold` | Seconds eyes must be closed before alarm triggers | `0.25` |
| `interval` | Check interval in seconds (only used when preview is disabled) | `0.5` |
| `show_preview` | Show camera preview window with detection overlay | `true` |
| `ear_threshold` | Eye Aspect Ratio threshold - lower values mean eyes must be more closed to trigger | `0.2` |

## Sharing the App

### Option 1: Zip the App Bundle

1. Build the app using `python build.py`

2. Create a zip file:
   ```bash
   cd dist
   zip -r DoomScrollingAlarm.zip DoomScrollingAlarm.app
   ```

3. Share the `DoomScrollingAlarm.zip` file

**Important for recipients:**
- After downloading, extract the zip
- Right-click the app and select "Open" to bypass Gatekeeper on first launch
- If macOS says the app is damaged, run this in Terminal:
  ```bash
  xattr -cr /path/to/DoomScrollingAlarm.app
  ```

### Option 2: Create a DMG (Disk Image)

1. Build the app using `python build.py`

2. Create a DMG:
   ```bash
   hdiutil create -volname "DoomScrollingAlarm" -srcfolder dist/DoomScrollingAlarm.app -ov -format UDZO DoomScrollingAlarm.dmg
   ```

3. Share the `DoomScrollingAlarm.dmg` file

### Note on Code Signing

The built app is ad-hoc signed, which means:
- It will work on your machine without issues
- Recipients may see security warnings from macOS Gatekeeper
- For distribution without warnings, you need an Apple Developer account ($99/year) to properly sign and notarize the app

## Controls

- Press `q` in the preview window to quit
- Close your eyes for longer than the threshold to trigger the alarm
- Open your eyes to stop the alarm

## Troubleshooting

### App won't open / "App is damaged"
```bash
xattr -cr /path/to/DoomScrollingAlarm.app
```

### Camera not detected
- Ensure no other app is using the camera
- Check System Preferences > Privacy & Security > Camera

### Alarm video doesn't play with sound
- The app tries to use `ffplay` (from FFmpeg) for video playback
- Install FFmpeg: `brew install ffmpeg`
- Falls back to QuickTime Player if ffplay is unavailable

### Eyes detected as closed when open
- Adjust `ear_threshold` in config.json (try increasing to 0.25 or 0.3)
- Ensure good lighting on your face
- Position camera at eye level

## License

MIT License
