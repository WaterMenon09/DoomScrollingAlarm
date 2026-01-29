#!/usr/bin/env python3
"""Build script to create standalone macOS app using PyInstaller."""

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
APP_NAME = "DoomScrollingAlarm"
# Minimum macOS version to support (11.0 = Big Sur)
MIN_MACOS_VERSION = "11.0"


def create_icns_from_png(png_path: Path, icns_path: Path):
    """Convert a PNG image to macOS .icns format."""
    iconset_path = BASE_DIR / "AppIcon.iconset"

    # Clean up any existing iconset
    if iconset_path.exists():
        shutil.rmtree(iconset_path)
    iconset_path.mkdir()

    # Required icon sizes for macOS
    sizes = [16, 32, 64, 128, 256, 512]

    for size in sizes:
        # Standard resolution
        output = iconset_path / f"icon_{size}x{size}.png"
        subprocess.run([
            "sips", "-z", str(size), str(size),
            str(png_path), "--out", str(output)
        ], capture_output=True)

        # Retina resolution (@2x)
        retina_size = size * 2
        if retina_size <= 1024:
            output_2x = iconset_path / f"icon_{size}x{size}@2x.png"
            subprocess.run([
                "sips", "-z", str(retina_size), str(retina_size),
                str(png_path), "--out", str(output_2x)
            ], capture_output=True)

    # Convert iconset to icns
    subprocess.run([
        "iconutil", "-c", "icns", str(iconset_path), "-o", str(icns_path)
    ], check=True)

    # Clean up iconset
    shutil.rmtree(iconset_path)
    print(f"Created app icon: {icns_path}")


def update_info_plist():
    """Add camera permission description to Info.plist."""
    plist_path = BASE_DIR / "dist" / f"{APP_NAME}.app" / "Contents" / "Info.plist"

    if not plist_path.exists():
        return

    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)

    plist["NSCameraUsageDescription"] = "This app uses the camera to detect if your eyes are open or closed."
    plist["CFBundleName"] = APP_NAME
    plist["CFBundleDisplayName"] = "Doom Scrolling Alarm"
    plist["CFBundleShortVersionString"] = "1.0.0"
    plist["LSMinimumSystemVersion"] = MIN_MACOS_VERSION

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    print("Updated Info.plist with camera permission.")


def build():
    # Ensure model is downloaded first
    print("Ensuring model is downloaded...")
    subprocess.run([sys.executable, "-c",
        "from eye_tracker import load_config, download_model; download_model(load_config())"],
        cwd=BASE_DIR, check=True)

    # Create app icon from logo.png
    logo_path = BASE_DIR / "logo.png"
    icon_path = BASE_DIR / "AppIcon.icns"
    if logo_path.exists():
        create_icns_from_png(logo_path, icon_path)
    else:
        icon_path = None
        print("Warning: logo.png not found, building without custom icon")

    # Find mediapipe location
    result = subprocess.run(
        [sys.executable, "-c", "import mediapipe; print(mediapipe.__path__[0])"],
        capture_output=True, text=True, check=True
    )
    mediapipe_path = result.stdout.strip()

    print(f"Building macOS app (targeting macOS {MIN_MACOS_VERSION}+)...")

    # Set deployment target for older macOS compatibility
    env = os.environ.copy()
    env["MACOSX_DEPLOYMENT_TARGET"] = MIN_MACOS_VERSION

    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--windowed",  # Create .app bundle
        "--onedir",  # Directory bundle
        "--noconfirm",  # Overwrite existing build
    ]

    # Add icon if available
    if icon_path and icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    cmd.extend([
        # Add data files
        "--add-data", "config.json:.",
        "--add-data", "alarm:alarm",
        "--add-data", "models:models",

        # Include all of mediapipe (including native libraries)
        "--add-data", f"{mediapipe_path}:mediapipe",

        # Collect all mediapipe submodules
        "--collect-all", "mediapipe",

        # Hidden imports
        "--hidden-import", "mediapipe",
        "--hidden-import", "mediapipe.tasks",
        "--hidden-import", "mediapipe.tasks.c",
        "--hidden-import", "mediapipe.tasks.python",
        "--hidden-import", "mediapipe.tasks.python.vision",
        "--hidden-import", "mediapipe.tasks.python.core",

        # Entry point
        "main.py"
    ])

    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)

    # Add camera permission to Info.plist
    update_info_plist()

    # Re-sign the app after modifying Info.plist
    app_path = BASE_DIR / "dist" / f"{APP_NAME}.app"
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app_path)], check=True)
    print("Re-signed the app.")

    # Remove quarantine attributes to prevent Gatekeeper cancel icon
    subprocess.run(["xattr", "-cr", str(app_path)], check=True)
    print("Removed quarantine attributes.")

    print(f"\nBuild complete!")
    print(f"App location: {app_path}")
    print(f"\nTo run: open dist/{APP_NAME}.app")
    print(f"To distribute: zip the dist/{APP_NAME}.app folder")


if __name__ == "__main__":
    build()
