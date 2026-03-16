import os
import shutil
import yaml
from pathlib import Path

# === НАСТРОЙКИ (проверьте пути!) ===
INPUT_DIR = 'All_Datasets'  # Папка, где лежат ваши 7 папок с датасетами
OUTPUT_DIR = 'Master_Dataset'  # Папка, куда скрипт сложит всё вместе

# Финальный список классов (порядок важен, он станет номерами 0, 1, 2, 3...)
MASTER_CLASSES = ['phone', 'laptop', 'ping_pong_ball', 'pen', 'person', 'book', 'backpack']


def make_dirs():
    # Создаем пустые папки для финального датасета
    for split in ['train', 'valid', 'test']:
        Path(f"{OUTPUT_DIR}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/{split}/labels").mkdir(parents=True, exist_ok=True)


def main():
    make_dirs()
    image_counter = 0  # Счетчик для уникальных имен файлов

    # 1. Перебираем все папки внутри All_Datasets (phone, laptop и т.д.)
    for dataset_folder in os.listdir(INPUT_DIR):
        ds_path = os.path.join(INPUT_DIR, dataset_folder)
        yaml_file = os.path.join(ds_path, 'data.yaml')

        if not os.path.isdir(ds_path) or not os.path.exists(yaml_file):
            continue

        print(f"\nОбработка датасета: {dataset_folder}")

        # 2. Читаем старый data.yaml
        with open(yaml_file, 'r', encoding='utf-8') as f:
            old_yaml = yaml.safe_load(f)

        old_names = old_yaml.get('names', [])
        if isinstance(old_names, dict):
            old_names = [old_names[k] for k in sorted(old_names.keys())]

        # 3. Составляем карту перевода старых ID в новые ID
        id_mapping = {}
        for old_id, name in enumerate(old_names):
            name_lower = str(name).lower().strip()
            new_id = -1

            # Умный поиск нужного класса (например, bag станет backpack)
            for i, master_name in enumerate(MASTER_CLASSES):
                if master_name.lower() in name_lower or name_lower in master_name.lower():
                    new_id = i;
                    break
                elif 'bag' in name_lower and master_name == 'backpack':
                    new_id = i;
                    break
                elif 'people' in name_lower and master_name == 'person':
                    new_id = i;
                    break

            if new_id != -1:
                id_mapping[old_id] = new_id
                print(f"  [+] Класс '{name}' (old_id: {old_id}) -> '{MASTER_CLASSES[new_id]}' (new_id: {new_id})")
            else:
                print(f"  [-] Игнорируем класс '{name}', так как он не нужен по заданию.")

        if not id_mapping:
            continue

        # 4. Копируем картинки и переписываем txt файлы
        for split in ['train', 'valid', 'test']:
            split_img_dir = os.path.join(ds_path, split, 'images')
            split_lbl_dir = os.path.join(ds_path, split, 'labels')

            if not os.path.exists(split_img_dir) or not os.path.exists(split_lbl_dir):
                continue

            for img_file in os.listdir(split_img_dir):
                name_without_ext = os.path.splitext(img_file)[0]
                txt_file = name_without_ext + '.txt'

                old_txt_path = os.path.join(split_lbl_dir, txt_file)
                old_img_path = os.path.join(split_img_dir, img_file)

                if not os.path.exists(old_txt_path): continue

                # Читаем старые метки
                with open(old_txt_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    old_class_id = int(parts[0])

                    # Меняем ID и записываем новую строчку
                    if old_class_id in id_mapping:
                        new_class_id = id_mapping[old_class_id]
                        new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)

                # Сохраняем картинку и файл txt с уникальным именем
                if new_lines:
                    new_base_name = f"image_{image_counter:05d}"
                    new_img_ext = os.path.splitext(img_file)[1]

                    new_img_path = os.path.join(OUTPUT_DIR, split, 'images', new_base_name + new_img_ext)
                    new_txt_path = os.path.join(OUTPUT_DIR, split, 'labels', new_base_name + '.txt')

                    shutil.copy2(old_img_path, new_img_path)
                    with open(new_txt_path, 'w') as f:
                        f.writelines(new_lines)

                    image_counter += 1

    # 5. Создаем финальный data.yaml для YOLOv8
    master_yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(master_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(MASTER_CLASSES),
            'names': MASTER_CLASSES
        }, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ ГОТОВО! Объединенный датасет лежит в папке: {OUTPUT_DIR}")
    print(f"✅ Всего сохранено картинок: {image_counter}")


if __name__ == '__main__':
    main()