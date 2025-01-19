import sys
import csv
import copy
import itertools
from collections import Counter
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QWidget, QFrame,
                             QDesktopWidget, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QRect

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

class StyledButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

class HandGestureRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Voice")

        # Fixed window and video size
        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 900
        self.VIDEO_WIDTH = 1280
        self.VIDEO_HEIGHT = 720

        self.setGeometry(100, 100, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # Set overall style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTextEdit {
                background-color: white;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                font-size: 18px;
                padding: 10px;
            }
        """)

        # Central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Video and Detection Frame
        self.video_frame = QLabel()
        self.video_frame.setFixedSize(self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.video_frame.setStyleSheet("""
            QLabel {
                border: 3px solid #4CAF50;
                border-radius: 10px;
                background-color: black;
            }
        """)

        # Add drop shadow to video frame
        video_shadow = QGraphicsDropShadowEffect()
        video_shadow.setBlurRadius(20)
        video_shadow.setColor(QColor(0, 0, 0, 80))
        video_shadow.setOffset(0, 0)
        self.video_frame.setGraphicsEffect(video_shadow)

        # Sentence Builder Frame
        sentence_layout = QVBoxLayout()
        sentence_layout.setSpacing(15)

        self.sentence_display = QTextEdit()
        self.sentence_display.setMinimumHeight(200)

        # Styled buttons
        self.copy_button = StyledButton("Copy Sentence")
        self.copy_button.clicked.connect(self.copy_sentence)

        self.clear_button = StyledButton("Clear Sentence")
        self.clear_button.clicked.connect(self.clear_sentence)

        # Labels
        title_label = QLabel("Hand Gesture Sentence Builder")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 10px;
            }
        """)

        sentence_layout.addWidget(title_label)
        sentence_layout.addWidget(self.sentence_display)
        sentence_layout.addWidget(self.copy_button)
        sentence_layout.addWidget(self.clear_button)

        # Add frames to main layout
        main_layout.addWidget(self.video_frame, 2)  # Give more space to video
        main_layout.addLayout(sentence_layout, 1)

        # Hand Gesture Recognition Setup
        self.setup_hand_recognition()

        # Sentence building
        self.current_sentence = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_delay = 2 #Delay in second

        # Detection box coordinates (center of frame)
        self.detection_box = QRect(self.VIDEO_WIDTH // 4, self.VIDEO_HEIGHT // 4,
                                    self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT // 2)

    def setup_hand_recognition(self):
        # Camera and MediaPipe setup
        self.cap = cv.VideoCapture(0)

        # Set EXACT camera resolution
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.VIDEO_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.VIDEO_HEIGHT)

        # Verify resolution
        actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual Camera Resolution: {actual_width}x{actual_height}")

        # MediaPipe Hands setup
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Classifiers
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Load labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        # Timer for video capture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms interval

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip the frame horizontally
        frame = cv.flip(frame, 1)

        # Convert the frame from BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Draw detection box with fixed green color (now on the RGB frame)
        cv.rectangle(frame_rgb,
                     (self.detection_box.x(), self.detection_box.y()),
                     (self.detection_box.x() + self.detection_box.width(),
                      self.detection_box.y() + self.detection_box.height()),
                     (0, 255, 0), 2)

        # Add the "INPUT AREA" text (now on the RGB frame)
        text_x = self.detection_box.x() + 10
        text_y = self.detection_box.y() - 10
        cv.putText(frame_rgb, "INPUT AREA", (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)


        # Hand detection
        results = self.hands.process(frame_rgb)

        hand_in_box = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Process landmarks
                landmark_list = self.calc_landmark_list(frame_rgb, hand_landmarks)

                # Check if hand is FULLY in detection box
                hand_in_box = self.is_hand_fully_in_detection_box(landmark_list)

                if hand_in_box:
                    pre_processed_landmark_list = self.pre_process_landmark(landmark_list)

                    # Classify hand sign
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    gesture = self.keypoint_classifier_labels[hand_sign_id]

                    # Update sentence logic
                    self.update_sentence(gesture)

                    # Draw landmarks with fixed blue color (on the RGB frame)
                    self.draw_landmarks(frame_rgb, landmark_list)

        # Convert the processed frame to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Fixed size scaling to match the video frame size
        scaled_pixmap = pixmap.scaled(
            self.VIDEO_WIDTH, self.VIDEO_HEIGHT,
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation
        )

        self.video_frame.setPixmap(scaled_pixmap)
        self.video_frame.setAlignment(Qt.AlignCenter)

    def is_hand_fully_in_detection_box(self, landmark_list):
        # Check if ALL landmarks are inside the detection box
        for landmark in landmark_list:
            if not (self.detection_box.x() <= landmark[0] <= self.detection_box.x() + self.detection_box.width() and
                    self.detection_box.y() <= landmark[1] <= self.detection_box.y() + self.detection_box.height()):
                return False
        return True

    def update_sentence(self, gesture):
        # Sentence building logic with delay
        current_time = time.time()
        if gesture != self.last_gesture:
            if (current_time - self.last_gesture_time) >= self.gesture_delay:
              if gesture != 'None':
                  self.current_sentence.append(gesture)
                  self.sentence_display.setText(' '.join(self.current_sentence))
              self.last_gesture = gesture
              self.last_gesture_time = current_time

    def copy_sentence(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.sentence_display.toPlainText())

    def clear_sentence(self):
        self.current_sentence = []
        self.sentence_display.clear()
        self.last_gesture = None

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

        for landmark_point in temp_landmark_list:
            landmark_point[0] -= base_x
            landmark_point[1] -= base_y

        # Convert to one-dimensional list and normalize
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        return list(map(normalize_, temp_landmark_list))

    def draw_landmarks(self, image, landmark_point):
        # Simple landmark drawing
        for index, landmark in enumerate(landmark_point):
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0), -1)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = HandGestureRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()