# Traffic Vehicle Detection using YOLOv8

 # Overview:

This project uses the YOLOv8 model to detect and count vehicles in a traffic video in real-time. It uses OpenCV to process video frames and PyTorch for running the model efficiently.

 # Features:

1. Detects vehicles in real-time

2. Draws bounding boxes around detected vehicles

3. Displays the number of vehicles in each frame

4. Uses GPU (if available) for faster processing

  # How It Works:

1. Loads the YOLOv8 model (yolov8n.pt).

2. Reads frames from trafficVid.mp4.

3. Resizes frames to 640x480 pixels.

4. Detects vehicles and draws bounding boxes.

5. Counts the number of vehicles in each frame.

6. Press q to stop the video.

  # Output Example:

The program will display the video with boxes around detected vehicles and the vehicle count on the screen.

#  Requirements:

1. Python 3.8+

2. PyTorch

3. OpenCV

4. Ultralytics YOLOv8
