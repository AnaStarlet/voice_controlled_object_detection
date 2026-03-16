import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QComboBox, QHBoxLayout,
                             QGroupBox, QTextEdit, QFileDialog, QMessageBox,
                             QSlider, QProgressBar, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import os
from datetime import datetime
from voice_control import VoiceController
from ultralytics import YOLO

PINK_STYLE = """
QMainWindow {
    background-color: #FFF0F5;
}
QGroupBox {
    background-color: #FFE4E1;
    border: 2px solid #FFB6C1;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: bold;
    color: #8B5F65;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 10px 0 10px;
    background-color: #FFE4E1;
    color: #8B5F65;
}
QPushButton {
    background-color: #FFB6C1;
    border: 2px solid #FFA5B8;
    border-radius: 8px;
    padding: 8px 15px;
    color: #4A3F40;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #FFA5B8;
}
QPushButton:pressed {
    background-color: #FF94A5;
}
QLabel {
    color: #6B4E4E;
    font-size: 12px;
}
QComboBox {
    background-color: #FFE4E1;
    border: 2px solid #FFB6C1;
    border-radius: 5px;
    padding: 5px;
    color: #6B4E4E;
}
QProgressBar {
    border: 2px solid #FFB6C1;
    border-radius: 5px;
    text-align: center;
    color: #6B4E4E;
    background-color: #FFE4E1;
}
QProgressBar::chunk {
    background-color: #FFB6C1;
    border-radius: 3px;
}
QTextEdit {
    background-color: #FFE4E1;
    border: 2px solid #FFB6C1;
    border-radius: 8px;
    color: #6B4E4E;
}
QSlider::groove:horizontal {
    border: 1px solid #FFB6C1;
    height: 6px;
    background: #FFE4E1;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFB6C1;
    border: 2px solid #FF94A5;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
#startButton {
    background-color: #FF69B4;
    border: 2px solid #FF1493;
    color: white;
    font-size: 14px;
    padding: 12px;
}
#startButton:hover {
    background-color: #FF82B4;
}
#stopButton {
    background-color: #FFA07A;
    border: 2px solid #FF7F50;
    color: white;
    font-size: 14px;
    padding: 12px;
}
#stopButton:hover {
    background-color: #FFB48A;
}
"""

# КЛАССЫ ОБЪЕКТОВ - простые русские названия
CLASSES = ['pen', 'phone', 'laptop', 'ball', 'person', 'book', 'bag']
CLASSES_RU = ['Ручка', 'Телефон', 'Ноутбук', 'Мяч', 'Человек', 'Книга', 'Сумка']

# Цвета для рамок
COLORS = [
    (255, 105, 180),  # розовый - ручка
    (255, 182, 193),  # светло-розовый - телефон
    (255, 160, 122),  # коралловый - ноутбук
    (255, 218, 185),  # персиковый - мяч
    (255, 228, 225),  # бело-розовый - человек
    (255, 240, 245),  # лавандовый - книга
    (255, 228, 196),  # бисквитный - сумка
]


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, float, dict, str)
    fps_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)

    def __init__(self, camera_id=0, model_path=None):
        super().__init__()
        self._run_flag = True
        self.camera_id = camera_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fps = 0
        self.current_target = None

        # Загружаем YOLO модель
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        self.colors = COLORS

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.log_signal.emit(f"✅ Модель загружена: {os.path.basename(path)}")
        except Exception as e:
            self.log_signal.emit(f"❌ Ошибка загрузки: {e}")

    def set_target(self, target):
        self.current_target = target
        self.log_signal.emit(f"🎯 Ищем: {target}")

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.log_signal.emit("❌ Не удалось открыть камеру")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        fps_timer = time.time()
        self.log_signal.emit("📷 Камера запущена")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, found_objects = self.detect_objects_yolo(frame)

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = frame_count
                self.fps_signal.emit(self.fps)
                frame_count = 0
                fps_timer = time.time()

            self.change_pixmap_signal.emit(processed_frame, self.fps,
                                           found_objects, self.current_target)

        cap.release()
        self.log_signal.emit("📷 Камера остановлена")

    def detect_objects_yolo(self, frame):
        result_frame = frame.copy()
        found = {}

        if self.model is None:
            cv2.putText(result_frame, "Model not loaded", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame, found

        results = self.model(frame, verbose=False)[0]
        target_found = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            if class_name in CLASSES:
                class_idx = CLASSES.index(class_name)
                color = self.colors[class_idx]
                english_name = class_name.capitalize()

                # Словарь для красивых названий
                display_names = {
                    'pen': 'Pen',
                    'phone': 'Phone',
                    'laptop': 'Laptop',
                    'ball': 'Ball',
                    'person': 'Person',
                    'book': 'Book',
                    'bag': 'Bag'
                }
                display_name = display_names.get(class_name, english_name)

                if self.current_target and class_name == self.current_target:
                    color = (0, 255, 0)
                    target_found = True
                    found['target'] = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    }

                if class_name not in found:
                    found[class_name] = {
                        'count': 1,
                        'max_conf': conf
                    }
                else:
                    found[class_name]['count'] += 1
                    found[class_name]['max_conf'] = max(found[class_name]['max_conf'], conf)

                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

                # Название на английском
                label = f"{display_name} {int(conf * 100)}%"
                cv2.putText(result_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.current_target:
            display_names = {
                'pen': 'Pen',
                'phone': 'Phone',
                'laptop': 'Laptop',
                'ball': 'Ball',
                'person': 'Person',
                'book': 'Book',
                'bag': 'Bag'
            }
            target_display = display_names.get(self.current_target, self.current_target.capitalize())
            target_text = f"Looking for: {target_display}"

            if target_found:
                target_text = f"{target_display} FOUND!"
                color = (0, 255, 0)
            else:
                color = (255, 105, 180)

            cv2.putText(result_frame, target_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result_frame, found
    def stop(self):
        self._run_flag = False
        self.wait()


class ObjectSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌸 ПОИСК ОБЪЕКТОВ ПО ГОЛОСУ 🌸")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet(PINK_STYLE)

        self.thread = None
        self.voice = VoiceController()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        main_layout.addWidget(self.create_video_panel(), 2)
        main_layout.addWidget(self.create_control_panel(), 1)

        self.voice.start_listening(self.on_voice_command)

    def create_video_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)

        title = QLabel("🌸 ВИДЕОПОТОК 🌸")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 3px solid #FFB6C1; border-radius: 15px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("🌸 Камера не активна 🌸")
        layout.addWidget(self.video_label)

        self.command_label = QLabel("🎤 Скажите: 'найди ручку'")
        self.command_label.setFont(QFont("Arial", 12))
        self.command_label.setAlignment(Qt.AlignCenter)
        self.command_label.setStyleSheet("color: #FF69B4;")
        layout.addWidget(self.command_label)

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 12))
        self.fps_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.fps_label)

        return panel

    def create_control_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)

        # Камера
        camera_group = QGroupBox("📷 КАМЕРА")
        camera_layout = QVBoxLayout()
        self.camera_combo = QComboBox()
        self.scan_cameras()
        camera_layout.addWidget(self.camera_combo)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Модель
        model_group = QGroupBox("🧠 МОДЕЛЬ")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.scan_models()
        model_layout.addWidget(self.model_combo)

        load_btn = QPushButton("📂 Загрузить")
        load_btn.clicked.connect(self.load_custom_model)
        model_layout.addWidget(load_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Кнопка запуска
        self.start_btn = QPushButton("▶ ЗАПУСТИТЬ")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.toggle_recognition)
        layout.addWidget(self.start_btn)

        # Текущий объект
        target_group = QGroupBox("🎯 ИЩЕМ ОБЪЕКТ")
        target_layout = QVBoxLayout()
        self.target_label = QLabel("—")
        self.target_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setStyleSheet("color: #FF69B4;")
        target_layout.addWidget(self.target_label)
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Лог
        log_group = QGroupBox("📝 ЛОГ")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Список команд
        commands_group = QGroupBox("🎤 ГОЛОСОВЫЕ КОМАНДЫ")
        commands_layout = QVBoxLayout()
        commands = [
            "• 'найди ручку'",
            "• 'найди телефон'",
            "• 'найди ноутбук'",
            "• 'найди мяч'",
            "• 'найди человека'",
            "• 'найди книгу'",
            "• 'найди сумку'"
        ]
        for cmd in commands:
            label = QLabel(cmd)
            label.setStyleSheet("color: #8B5F65;")
            commands_layout.addWidget(label)
        commands_group.setLayout(commands_layout)
        layout.addWidget(commands_group)

        self.scan_cameras()
        self.scan_models()
        self.log("🌸 Приложение запущено")

        return panel

    def on_voice_command(self, text):
        self.log(f"🎤 Распознано: {text}")
        self.command_label.setText(f"🎤 Сказано: {text}")

        target = self.voice.parse_command(text)
        if target:
            self.log(f"🎯 Найден объект: {target}")
            self.target_label.setText(CLASSES_RU[CLASSES.index(target)])

            if self.thread:
                self.thread.set_target(target)
        else:
            self.log("❌ Объект не распознан")

    def scan_cameras(self):
        self.camera_combo.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Камера {i}")
                cap.release()

    def scan_models(self):
        self.model_combo.clear()
        models = [f for f in os.listdir('.') if f.endswith('.pt')]
        if models:
            self.model_combo.addItems(models)

    def load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "", "Model files (*.pt *.pth)")
        if path:
            model_name = os.path.basename(path)
            self.model_combo.addItem(model_name)
            self.model_combo.setCurrentText(model_name)
            self.log(f"📂 Модель загружена: {model_name}")

    def toggle_recognition(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
            self.start_btn.setText("▶ ЗАПУСТИТЬ")
            self.start_btn.setObjectName("startButton")
            self.video_label.setText("🌸 Камера не активна 🌸")
            self.log("⏹ Распознавание остановлено")
        else:
            model_name = self.model_combo.currentText()
            if not model_name:
                QMessageBox.warning(self, "Ошибка", "Выберите модель!")
                return

            camera_id = self.camera_combo.currentIndex()

            self.thread = VideoThread(camera_id, model_name)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.fps_signal.connect(self.update_fps)
            self.thread.log_signal.connect(self.log)

            self.thread.start()
            self.start_btn.setText("⏹ ОСТАНОВИТЬ")
            self.start_btn.setObjectName("stopButton")
            self.log(f"▶ Распознавание запущено (модель: {model_name})")

    def update_image(self, cv_img, fps, objects, target):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")

    def log(self, msg):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{time_str}] {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        self.voice.stop_listening()
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ObjectSearchApp()
    window.show()
    sys.exit(app.exec_())