#!/usr/bin/env python3
"""Main entry point for the Eye Tracking Alarm app."""

import json
from eye_tracker import load_config, EyeTracker


def main():
    config = load_config()
    print(f"Loaded config: {json.dumps(config, indent=2)}")

    tracker = EyeTracker(config)
    tracker.run(show_preview=config.get("show_preview", True))


if __name__ == "__main__":
    main()
