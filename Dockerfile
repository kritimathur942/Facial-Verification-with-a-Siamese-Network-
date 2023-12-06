# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Set environment variable for video file location
ENV VIDEO_FILE="C:\\Users\\praya\\OneDrive\\Desktop\\emotion-detection--1\\video.mp4"

# Download the face detection model
RUN mkdir -p model
RUN wget -O model/haarcascade_frontalface_default.xml https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

# Define the command to run your application
CMD [ "python", "TestEmotioDetector.py" ]
