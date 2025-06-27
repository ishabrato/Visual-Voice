import sys
import csv
import copy
import itertools
from collections import deque, Counter
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QWidget, QFrame,
                             QDesktopWidget, QGraphicsDropShadowEffect, QStatusBar,
                             QDialog, QListWidget, QAbstractItemView, QListWidgetItem, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc, camera_utils


class CameraSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()

        self.camera_list_widget = QListWidget()
        self.camera_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.camera_list_widget)

        self.populate_camera_list()

        save_button = QPushButton("Save Selection")
        save_button.clicked.connect(self.save_selection)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def populate_camera_list(self):
        self.available_cameras = camera_utils.get_available_cameras()
        current_camera_index = camera_utils.load_camera_setting()

        if not self.available_cameras:
            self.camera_list_widget.addItem("No cameras found.")
            self.camera_list_widget.setEnabled(False)
            return

        for cam_index in self.available_cameras:
            item = QListWidgetItem(f"Camera {cam_index}")
            self.camera_list_widget.addItem(item)
            if cam_index == current_camera_index:
                item.setSelected(True)

    def save_selection(self):
        selected_items = self.camera_list_widget.selectedItems()
        if selected_items:
            selected_index_str = selected_items[0].text().replace("Camera ", "")
            selected_index = int(selected_index_str)
            camera_utils.save_camera_setting(selected_index)
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a camera.")


class StyledButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; border: none;
                padding: 10px 20px; font-size: 16px; border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)

class HandGestureRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Voice")

        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 900
        self.VIDEO_WIDTH = 1280
        self.VIDEO_HEIGHT = 720

        self.setGeometry(100, 100, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setStyleSheet("QMainWindow { background-color: #f0f0f0; }")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        self.setCentralWidget(central_widget)

        header = QLabel("✋ Visual Voice – Real-time Hand Gesture Recognition")
        header.setStyleSheet("""
            QLabel {
                font-size: 28px; font-weight: bold; color: #333;
                padding: 15px; border-radius: 8px; background-color: #4CAF50;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        self.video_frame = QLabel()
        self.video_frame.setFixedSize(self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        self.video_frame.setStyleSheet("QLabel { border: 3px solid #4CAF50; border-radius: 10px; background-color: black; }")

        video_shadow = QGraphicsDropShadowEffect()
        video_shadow.setBlurRadius(20)
        video_shadow.setColor(QColor(0, 0, 0, 80))
        video_shadow.setOffset(0, 0)
        self.video_frame.setGraphicsEffect(video_shadow)

        sentence_layout = QVBoxLayout()
        sentence_layout.setContentsMargins(20, 20, 20, 20)
        sentence_layout.setSpacing(20)

        title_label = QLabel("Your Sentence")
        title_label.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; color: #4CAF50; }")

        self.sentence_display = QTextEdit()
        self.sentence_display.setReadOnly(True)
        self.sentence_display.setMinimumHeight(200)
        self.sentence_display.setStyleSheet("""
            QTextEdit {
                background-color: white; border: 2px solid #4CAF50; border-radius: 10px;
                font-size: 22px; padding: 15px; color: #333;
            }
        """)

        self.copy_button = StyledButton("Copy Sentence")
        self.copy_button.clicked.connect(self.copy_sentence)
        self.clear_button = StyledButton("Clear Sentence")
        self.clear_button.clicked.connect(self.clear_sentence)
        self.backspace_button = StyledButton("Backspace")
        self.backspace_button.clicked.connect(self.backspace_sentence)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.backspace_button)
        button_layout.addWidget(self.clear_button)

        settings_button = StyledButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        button_layout.addWidget(settings_button)

        sentence_layout.addWidget(title_label)
        sentence_layout.addWidget(self.sentence_display)
        sentence_layout.addLayout(button_layout)

        content_layout.addWidget(self.video_frame, 2)
        content_layout.addLayout(sentence_layout, 1)
        main_layout.addLayout(content_layout)

        self.setup_hand_recognition()
        self.init_sentence_builder()

    def init_sentence_builder(self):
        self.current_sentence = []
        self.candidate_char = ""
        self.last_added_char_time = 0
        self.last_stable_gesture = None
        self.stable_gesture_start_time = None

        self.GESTURE_BUFFER_SIZE = 20
        self.STABILITY_THRESHOLD = 1
        self.CONFIRMATION_TIME = 1
        self.ADD_COOLDOWN = 1.5

        self.gesture_buffer = deque(maxlen=self.GESTURE_BUFFER_SIZE)
        self.sentence_display.setText("")
        self.status_bar.showMessage("Ready. Hold a gesture to begin.")

    def setup_hand_recognition(self):
        camera_index = camera_utils.load_camera_setting()
        self.cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index} in ui.py. Please check if camera is in use or drivers are installed.")
            return
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(1) # Give camera time to initialize
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.VIDEO_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.VIDEO_HEIGHT)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        self.SPACE_GESTURE = "SPACE"

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv.flip(frame, 1)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        current_gesture = "None"
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(image_rgb, hand_landmarks)
                landmark_list = self.calc_landmark_list(image_rgb, hand_landmarks)
                pre_processed_list = self.pre_process_landmark(landmark_list)

                reshaped_input = np.array(pre_processed_list, dtype=np.float32).reshape(1, 21, 2)

                # --- NEW: Get both the ID and confidence ---
                hand_sign_id, confidence = self.keypoint_classifier(reshaped_input)

                # --- NEW: Apply the confidence threshold ---
                CONFIDENCE_THRESHOLD = 0.85 # You can tune this value (e.g., 0.8, 0.9)

                if confidence > CONFIDENCE_THRESHOLD:
                    current_gesture = self.keypoint_classifier_labels[hand_sign_id]
                else:
                    current_gesture = "None"

                # --- NEW: Pass confidence score for display ---
                image_rgb = self.draw_bounding_rect(image_rgb, brect)
                image_rgb = self.draw_landmarks(image_rgb, landmark_list)
                image_rgb = self.draw_info_text(image_rgb, brect, handedness, current_gesture, confidence)

        self.process_gesture_for_sentence(current_gesture)

        if self.candidate_char:
            cv.putText(image_rgb, f"Candidate: {self.candidate_char}", (10, 60),
                       cv.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 6, cv.LINE_AA)
            cv.putText(image_rgb, f"Candidate: {self.candidate_char}", (10, 60),
                       cv.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        h, w, ch = image_rgb.shape
        qt_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_frame.setPixmap(QPixmap.fromImage(qt_image).scaled(self.VIDEO_WIDTH, self.VIDEO_HEIGHT, Qt.KeepAspectRatio))

    def process_gesture_for_sentence(self, gesture):
        if gesture != "None":
            self.gesture_buffer.append(gesture)
        else:
            self.last_stable_gesture = None
            self.stable_gesture_start_time = None
            self.candidate_char = ""
            return

        if len(self.gesture_buffer) < self.GESTURE_BUFFER_SIZE:
            return

        most_common_gesture, count = Counter(self.gesture_buffer).most_common(1)[0]

        if count / self.GESTURE_BUFFER_SIZE >= self.STABILITY_THRESHOLD:
            if most_common_gesture != self.last_stable_gesture:
                self.last_stable_gesture = most_common_gesture
                self.stable_gesture_start_time = time.time()
                self.candidate_char = most_common_gesture

            if self.stable_gesture_start_time and (time.time() - self.stable_gesture_start_time) > self.CONFIRMATION_TIME:
                if (time.time() - self.last_added_char_time) > self.ADD_COOLDOWN:

                    char_to_add = ' ' if self.last_stable_gesture == self.SPACE_GESTURE else self.last_stable_gesture
                    self.current_sentence.append(char_to_add)
                    self.sentence_display.setText("".join(self.current_sentence))

                    status_message = "Added: Space" if char_to_add == ' ' else f"Added: '{char_to_add}'"
                    self.statusBar().showMessage(status_message, 3000)

                    self.last_added_char_time = time.time()
                    self.stable_gesture_start_time = None
                    self.last_stable_gesture = None
                    self.candidate_char = ""
                    self.gesture_buffer.clear()
        else:
            self.last_stable_gesture = None
            self.stable_gesture_start_time = None
            self.candidate_char = ""

    def backspace_sentence(self):
        if self.current_sentence:
            self.current_sentence.pop()
            self.sentence_display.setText("".join(self.current_sentence))
            self.statusBar().showMessage("Backspace", 2000)

    def copy_sentence(self):
        QApplication.clipboard().setText(self.sentence_display.toPlainText())
        self.statusBar().showMessage("Sentence copied to clipboard!", 3000)

    def clear_sentence(self):
        self.init_sentence_builder()

    def open_settings(self):
        dialog = CameraSettingsDialog(self)
        if dialog.exec_():
            # If settings were saved, re-initialize camera with new setting
            self.setup_hand_recognition()

    def calc_landmark_list(self, image, landmarks):
        return [[min(int(lm.x * image.shape[1]), image.shape[1] - 1), min(int(lm.y * image.shape[0]), image.shape[0] - 1)] for lm in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = temp_landmark_list[0]
        for lm in temp_landmark_list:
            lm[0] -= base_x
            lm[1] -= base_y
        flat_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, flat_list), default=0)
        if max_value == 0: return [0.0] * len(flat_list)
        return [n / max_value for n in flat_list]

    def calc_bounding_rect(self, image, landmarks):
        landmark_array = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in landmarks.landmark], dtype=np.int32)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    # --- NEW: Updated to accept and display confidence ---
    def draw_info_text(self, image, brect, handedness, hand_sign_text, confidence):
        # Define text properties
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)      # Black

        info_text = f"{handedness.classification[0].label[0:]}: "
        if hand_sign_text != "None":
            info_text += f"{hand_sign_text} ({confidence:.2f})"

        # Get text size
        (text_width, text_height), baseline = cv.getTextSize(info_text, font, font_scale, font_thickness)

        # Draw background rectangle
        cv.rectangle(image, (brect[0], brect[1] - text_height - baseline - 10),
                     (brect[0] + text_width + 10, brect[1]), bg_color, -1)

        # Draw text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - baseline - 5),
                   font, font_scale, text_color, font_thickness, cv.LINE_AA)
        return image

    def draw_landmarks(self, image, landmark_point):
        if landmark_point:
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmark_point) and end_idx < len(landmark_point):
                    cv.line(image, tuple(landmark_point[start_idx]), tuple(landmark_point[end_idx]), (255, 255, 255), 2)
            for point in landmark_point:
                cv.circle(image, tuple(point), 5, (0, 255, 0), -1)
        return image

    def draw_bounding_rect(self, image, brect):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
        return image

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