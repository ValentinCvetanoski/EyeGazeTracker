Eye Gaze Tracker

This Python project uses a webcam to monitor the user's gaze in real-time. When the user looks away from the screen, a random video from a predefined list is played automatically. The video stops immediately when the user looks back at the monitor.

It leverages MediaPipe for accurate eye and iris tracking and FFmpeg/ffplay to play videos smoothly without OpenCV overhead. The project is optimized for performance by scaling down the camera feed and applying fast landmark detection.

Key Features:

Real-time gaze detection using MediaPipe.

Random video playback when looking away.

Immediate video termination upon looking back.

Lightweight and optimized for low CPU usage.

Configurable video list and detection sensitivity.

Tech Stack:

Python 3.10+

OpenCV

MediaPipe

FFmpeg / ffplay

Use Case:
Ideal for productivity applications, attention tracking, or playful “troll mode” video triggers.
