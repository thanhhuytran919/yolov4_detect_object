# yolov4_detect_object
# YOLOv4 Object Detection

This is a simple web application built with Flask and OpenCV that performs object detection using the YOLOv4 model. It allows users to upload an image and detect objects in it, highlighting them with bounding boxes.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [API Endpoint](#api-endpoint)
- [Contributing](#contributing)

## Features

- Upload an image.
- Detect objects in the uploaded image.
- Highlight detected objects with bounding boxes.
- Provides a simple web interface for object detection.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- Flask
- OpenCV (opencv-python-headless)
- Numpy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thanhhuytran919/yolov4_detect_object.git
   cd yolov4_detect_object
2. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Usage

1. Run the Flask application:
   ```bash
   python app.py
2. Access the application in your web browser by navigating to http://localhost:5000.
3. Upload an image and click the "Detect" button to detect objects in the image. Detected objects will be highlighted with bounding boxes, and the result will be displayed on the web page.
4. You should get images in folder /Model/Test images to detect. Image result will save in folder static.

### API Endpoint
  > You can also use the API endpoint to perform object detection. Send a POST request to /api/detect with the image file to detect objects in the image.
  > The API response will be in JSON format and include information about the detected objects.

### Contributing
If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request. Your contributions are greatly appreciated.
  
