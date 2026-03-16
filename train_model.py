"""
ОБУЧЕНИЕ YOLOv8 НА ОБЪЕДИНЕННОМ ДАТАСЕТЕ
"""

from ultralytics import YOLO
import torch
import os

# Проверка устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Используем устройство: {device}")

# Загружаем предобученную модель YOLOv8
model = YOLO('yolov8n.pt')  # yolov8n (nano) - самая легкая

# Обучение
results = model.train(
    data='Master_Dataset/data.yaml',  # путь к твоему конфигу
    epochs=15,                         # количество эпох
    imgsz=320,                          # размер картинок
    batch=32,                            # размер батча
    device=device,
    workers=4,
    patience=10,                         # early stopping
    save=True,                            # сохранять лучшую модель
    project='runs/train',                  # папка для сохранения
    name='emotion_search',                  # имя эксперимента
    exist_ok=True
)

metrics = model.val(data='Master_Dataset/data.yaml', split='test')
print(f"\n📊 Результаты на тесте:")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")

model.export(format='onnx')  # можно экспортировать в ONNX
torch.save(model.model.state_dict(), "potapchik_search_model.pth")
print(f"\n✅ Модель сохранена: potapchik_search_model.pth")