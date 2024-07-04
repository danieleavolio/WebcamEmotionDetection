import cv2
import numpy as np
import os
import argparse
from fer import FER
import matplotlib.pyplot as plt
import time


def initialize_face_detector(weights_path):
    """Initialize the face detector with the given weights."""
    return cv2.FaceDetectorYN.create(weights_path, "", (0, 0))


def initialize_emotion_detector():
    """Initialize the emotion detector."""
    return FER(mtcnn=True)


def draw_emotion_info(frame, emotions, box, colors):
    """Draw the dominant emotion and all emotions with confidence on the frame."""
    x, y, w, h = box

    dominant_emotion = max(emotions, key=emotions.get)
    dominant_confidence = emotions[dominant_emotion]

    # Draw the box around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), colors[dominant_emotion], 2)
    cv2.putText(frame, f"{dominant_emotion} {dominant_confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[dominant_emotion], 2)

    # Draw each emotion's confidence
    for i, (emotion, confidence) in enumerate(emotions.items()):
        cv2.putText(frame, f"{emotion} {confidence:.2f}", (0, 35 * i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[emotion], 2)

def draw_error_message(frame, message):
    """Draw an error message on the frame."""
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def plot_emotion_trend(time_stamps, dominant_emotions):
    """Save a graph showing the trend of dominant emotions over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_stamps, dominant_emotions, marker='o')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Dominant Emotion')
    plt.title('Dominant Emotion Trend Over Time')
    plt.yticks(range(7), ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry'])
    plt.grid(True)
    plt.savefig('emotion_trend.png')
    plt.show()




def main(video_analysis):
    # Define the colors for each emotion
    COLORS = {
        'angry': (0, 0, 255),  # Red
        'disgust': (88, 57, 39),  # Brown
        'fear': (0, 0, 0),  # Black
        'happy': (0, 255, 0),  # Green
        'sad': (255, 0, 0),  # Blue
        'surprise': (0, 255, 255),  # Yellow
        'neutral': (255, 255, 255)  # White
    }

    directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(directory, "weights/face_detection_yunet_2023mar.onnx")

    face_detector = initialize_face_detector(weights_path)
    emotion_detector = initialize_emotion_detector()

    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    dominant_emotions = []
    time_stamps = []

    start_time = time.time()
    next_capture_time = start_time + 1

    while True:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        try:
            analysis = emotion_detector.detect_emotions(frame)
            if analysis:
                box = analysis[0]['box']
                emotions = analysis[0]['emotions']
                draw_emotion_info(frame, emotions, box, COLORS)

                current_time = time.time()
                if video_analysis and current_time >= next_capture_time:
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_emotions.append(dominant_emotion)
                    time_stamps.append(current_time - start_time)
                    next_capture_time += 1
            else:
                draw_error_message(frame, "No face detected.")

        except Exception as e:
            print(f"Error: {e}")

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    if video_analysis:
        plot_emotion_trend(time_stamps, dominant_emotions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Emotion Detection using Webcam")
    parser.add_argument('--analyze_video', action='store_true', help='Analyze video and create emotion trend graph')
    args = parser.parse_args()

    main(args.analyze_video)
