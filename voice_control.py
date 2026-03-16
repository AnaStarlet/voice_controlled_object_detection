"""
ГОЛОСОВОЕ УПРАВЛЕНИЕ ДЛЯ ПОИСКА ОБЪЕКТОВ
"""

import json
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import threading
import time


class VoiceController:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.listening = False
        self.current_command = ""
        self.command_callback = None
        self.load_model()

    def load_model(self):
        """Загрузка Vosk модели"""
        try:
            # Укажите путь к распакованной модели
            model_path = "vosk-model-small-ru-0.22"
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            print("✅ Голосовая модель загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки голосовой модели: {e}")
            print("💡 Скачайте модель с https://alphacephei.com/vosk/models")

    def audio_callback(self, indata, frames, time, status):
        """Получение аудио с микрофона"""
        if status:
            print(f"Статус аудио: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def start_listening(self, callback=None):
        """Запуск прослушивания микрофона"""
        if not self.model:
            print("❌ Модель не загружена")
            return

        self.command_callback = callback
        self.listening = True

        # Запуск в отдельном потоке
        self.thread = threading.Thread(target=self._listen_thread)
        self.thread.daemon = True
        self.thread.start()
        print("🎤 Микрофон запущен")

    def _listen_thread(self):
        """Поток для обработки голоса"""
        with sd.RawInputStream(samplerate=16000, blocksize=8000,
                               device=None, dtype='int16',
                               channels=1, callback=self.audio_callback):

            while self.listening:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')

                    if text and self.command_callback:
                        self.command_callback(text)

    def stop_listening(self):
        """Остановка прослушивания"""
        self.listening = False
        print("🎤 Микрофон остановлен")

    def parse_command(self, text):
        """Анализ голосовой команды"""
        text = text.lower()

        # Словарь соответствия слов и классов
        commands = {
            'ручка': 'pen',
            'ручку': 'pen',
            'pen': 'pen',

            'телефон': 'phone',
            'phone': 'phone',

            'ноутбук': 'laptop',
            'компьютер': 'laptop',
            'laptop': 'laptop',

            'мяч': 'ball',
            'ball': 'ball',

            'человек': 'person',
            'person': 'person',

            'книга': 'book',
            'книгу': 'book',
            'book': 'book',

            'сумка': 'bag',
            'bag': 'bag',
        }

        # Ищем ключевые слова
        for word, class_name in commands.items():
            if word in text:
                return class_name

        return None