"""
dataset_builder.py
------------------
Automated Dataset Preparation, Augmentation, Splitting, and Metadata Generation.

Generates the standardized structure:
dataset/
├── train/
│   ├── Faculty_001/
│   ├── Faculty_002/
│   ├── Student_001/
│   ├── Student_002/
│   ├── Student_003/
│   └── Student_004/
├── validation/
│   └── ...
└── test/
    └── ...

unknown_test/
├── Visitor_A/
├── Visitor_B/
└── Visitor_C/

Outputs:
- class_names.json
- person_metadata.json
"""

import os
import json
import random
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

from config import (
    DATASET_DIR, KNOWN_FACES_DIR, UNKNOWN_TEST_DIR,
    CLASS_NAMES_PATH, METADATA_PATH, IMAGE_SIZE, MIN_LAPLACIAN_BLUR_VAR
)
from face_detector import FaceDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetBuilder")

# Standard Reference Profiles for Student & Faculty Metadata
BASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "FAC001": {
        "class_id": "Faculty_001",
        "person_id": "FAC001",
        "name": "Dr. Rajesh Iyer",
        "role": "Faculty",
        "department": "Computer Science & Engineering",
        "designation": "Professor & Head of Department",
        "employee_id": "FAC001",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Computer Networks", "room": "Lab 3"},
            {"day": "Tuesday", "time": "09:00 - 11:00 AM", "subject": "Image Processing & CV", "room": "CV Lab"},
            {"day": "Thursday", "time": "02:00 - 04:00 PM", "subject": "Computer Vision Capstone", "room": "Project Lab"}
        ]
    },
    "FAC002": {
        "class_id": "Faculty_002",
        "person_id": "FAC002",
        "name": "Dr. Sunita Mehra",
        "role": "Faculty",
        "department": "Electronics & Communication",
        "designation": "Associate Professor",
        "employee_id": "FAC002",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Signals & Systems", "room": "ECE-201"},
            {"day": "Tuesday", "time": "11:30 - 01:30 PM", "subject": "VLSI Design & Hardware Lab", "room": "VLSI Lab"},
            {"day": "Thursday", "time": "10:00 - 11:00 AM", "subject": "Embedded Systems", "room": "ECE-202"}
        ]
    },
    "STU001": {
        "class_id": "Student_001",
        "person_id": "STU001",
        "name": "Rahul Kumar",
        "role": "Student",
        "department": "Computer Science & Engineering",
        "roll_number": "21CS001",
        "class": "CSE-3A",
        "year": "3rd Year",
        "cgpa": "8.92",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Computer Networks", "room": "Lab 3", "faculty": "Dr. Rajesh Iyer"},
            {"day": "Tuesday", "time": "09:00 - 11:00 AM", "subject": "Image Processing Lab", "room": "CV Lab", "faculty": "Dr. Rajesh Iyer"},
            {"day": "Wednesday", "time": "10:00 - 11:00 AM", "subject": "Cloud Computing", "room": "LH-104", "faculty": "Prof. K. Mehta"}
        ]
    },
    "STU002": {
        "class_id": "Student_002",
        "person_id": "STU002",
        "name": "Priya Sharma",
        "role": "Student",
        "department": "Electronics & Communication",
        "roll_number": "21EC014",
        "class": "ECE-3B",
        "year": "3rd Year",
        "cgpa": "9.15",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Signals & Systems", "room": "ECE-201", "faculty": "Dr. Sunita Mehra"},
            {"day": "Tuesday", "time": "11:30 - 01:30 PM", "subject": "VLSI Design & Hardware Lab", "room": "VLSI Lab", "faculty": "Dr. Sunita Mehra"}
        ]
    },
    "STU003": {
        "class_id": "Student_003",
        "person_id": "STU003",
        "name": "Amit Verma",
        "role": "Student",
        "department": "Information Technology",
        "roll_number": "22IT033",
        "class": "IT-2A",
        "year": "2nd Year",
        "cgpa": "8.45",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Data Structures & Algorithms", "room": "IT-101", "faculty": "Prof. S. Joshi"},
            {"day": "Wednesday", "time": "02:00 - 04:00 PM", "subject": "DSA Lab", "room": "Software Lab 2", "faculty": "Prof. S. Joshi"}
        ]
    },
    "STU004": {
        "class_id": "Student_004",
        "person_id": "STU004",
        "name": "Sneha Patel",
        "role": "Student",
        "department": "Artificial Intelligence & Data Science",
        "roll_number": "21AI019",
        "class": "AI-3A",
        "year": "3rd Year",
        "cgpa": "9.40",
        "timetable": [
            {"day": "Monday", "time": "10:00 - 11:00 AM", "subject": "Machine Learning Algorithms", "room": "AI-301", "faculty": "Dr. R. Bannerjee"},
            {"day": "Tuesday", "time": "01:00 - 03:00 PM", "subject": "Deep Learning Studio Lab", "room": "GPU Server Lab", "faculty": "Dr. R. Bannerjee"}
        ]
    }
}


def apply_realistic_augmentation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies realistic photographic variation:
    - Slight rotation (-12 to +12 degrees)
    - Slight zoom/scale (0.92 to 1.08)
    - Small translation (up to 4% shift)
    - Brightness adjustment (0.82 to 1.22)
    - Contrast adjustment (0.85 to 1.18)
    - Horizontal mirroring (50% probability)
    - Subtle Gaussian noise or gentle blur to simulate CCTV distances
    """
    h, w = img_bgr.shape[:2]
    aug = img_bgr.copy()

    # 1. Horizontal Flip
    if random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 2. Affine Transformation: Rotation, Scale, Translation
    angle = random.uniform(-12, 12)
    scale = random.uniform(0.92, 1.08)
    tx = random.uniform(-0.04 * w, 0.04 * w)
    ty = random.uniform(-0.04 * h, 0.04 * h)

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Brightness and Contrast
    alpha = random.uniform(0.85, 1.18)  # Contrast
    beta = random.uniform(-22, 22)      # Brightness
    aug = np.clip(alpha * aug.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 4. Subtle distance blur / CCTV sensor noise (25% chance)
    if random.random() < 0.25:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), random.uniform(0.4, 0.9))

    return aug


def build_multiclass_dataset(
    known_dir: str = KNOWN_FACES_DIR,
    output_dir: str = DATASET_DIR,
    unknown_dir: str = UNKNOWN_TEST_DIR,
    target_samples_per_person: int = 60,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parses known_faces, standardizes classes, generates augmented samples,
    and organizes into train / validation / test splits.
    """
    detector = FaceDetector()
    random.seed(42)
    np.random.seed(42)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    for d in [train_dir, val_dir, test_dir, unknown_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Discover all candidate people from known_faces/
    known_files = [f for f in os.listdir(known_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    known_files.sort()

    registered_classes: Dict[str, List[np.ndarray]] = {}
    person_id_to_class: Dict[str, str] = {}
    person_metadata: Dict[str, Dict[str, Any]] = {}

    for fname in known_files:
        filepath = os.path.join(known_dir, fname)
        base = os.path.splitext(fname)[0]
        person_key = base.split("_")[0]  # e.g., FAC001, STU001

        meta = dict(BASE_PROFILES.get(person_key, {
            "class_id": f"Class_{person_key}",
            "person_id": person_key,
            "name": " ".join(base.split("_")[1:]) if len(base.split("_")) > 1 else person_key,
            "role": "Registered Person",
            "department": "Campus Security",
            "timetable": []
        }))

        class_name = meta.get("class_id", f"Person_{person_key}")
        person_id_to_class[person_key] = class_name
        person_metadata[class_name] = meta

        img = cv2.imread(filepath)
        if img is not None:
            # Crop/standardize to face resolution
            face_crop = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            if class_name not in registered_classes:
                registered_classes[class_name] = []
            registered_classes[class_name].append(face_crop)

    # Sort class names deterministically to guarantee consistent class indexing
    sorted_class_names = sorted(list(registered_classes.keys()))
    class_names_map: Dict[str, str] = {str(idx): cname for idx, cname in enumerate(sorted_class_names)}

    logger.info("Found %d registered classes: %s", len(sorted_class_names), sorted_class_names)

    # 2. Generate augmented variations and split across train, val, test
    total_images_generated = 0

    for class_idx_str, class_name in class_names_map.items():
        base_images = registered_classes[class_name]
        all_samples: List[np.ndarray] = list(base_images)

        # Augment to reach target_samples_per_person
        attempts = 0
        while len(all_samples) < target_samples_per_person and attempts < target_samples_per_person * 5:
            attempts += 1
            src_img = random.choice(base_images)
            aug_img = apply_realistic_augmentation(src_img)

            # Ensure augmented image meets size & blur quality standards
            blur_score = detector.calculate_blur_variance(aug_img)
            if blur_score >= MIN_LAPLACIAN_BLUR_VAR:
                resized_sample = cv2.resize(aug_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                all_samples.append(resized_sample)

        random.shuffle(all_samples)

        # Calculate split indices
        n = len(all_samples)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        train_imgs = all_samples[:n_train]
        val_imgs = all_samples[n_train:n_train + n_val]
        test_imgs = all_samples[n_train + n_val:]

        # Create target subfolders
        c_train_dir = os.path.join(train_dir, class_name)
        c_val_dir = os.path.join(val_dir, class_name)
        c_test_dir = os.path.join(test_dir, class_name)

        for d in [c_train_dir, c_val_dir, c_test_dir]:
            os.makedirs(d, exist_ok=True)

        for i, im in enumerate(train_imgs):
            cv2.imwrite(os.path.join(c_train_dir, f"{class_name}_train_{i:03d}.jpg"), im)
        for i, im in enumerate(val_imgs):
            cv2.imwrite(os.path.join(c_val_dir, f"{class_name}_val_{i:03d}.jpg"), im)
        for i, im in enumerate(test_imgs):
            cv2.imwrite(os.path.join(c_test_dir, f"{class_name}_test_{i:03d}.jpg"), im)

        total_images_generated += len(all_samples)
        logger.info(
            "Class [%s] (ID %s): %d total (%d train, %d val, %d test)",
            class_name, class_idx_str, len(all_samples), len(train_imgs), len(val_imgs), len(test_imgs)
        )

    # 3. Create UNKNOWN test dataset (impostors not in any training class)
    # Generates 3 distinct unknown visitor profiles for Task 11 open-world rejection testing
    unknown_profiles = ["Visitor_A", "Visitor_B", "Visitor_C"]
    for u_name in unknown_profiles:
        u_dir = os.path.join(unknown_dir, u_name)
        os.makedirs(u_dir, exist_ok=True)
        # Synthesize distinct synthetic face-like textures & patterns
        for k in range(25):
            # Create synthetic gradient face avatar with distinct color/pattern
            canvas = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
            col = (
                int(50 + 60 * np.sin(k * 0.4)),
                int(70 + 50 * np.cos(k * 0.3)),
                int(100 + 70 * np.sin(k * 0.7))
            )
            cv2.circle(canvas, (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), 42, col, -1)
            cv2.circle(canvas, (IMAGE_SIZE[0] // 2 - 14, IMAGE_SIZE[1] // 2 - 10), 8, (230, 230, 230), -1)
            cv2.circle(canvas, (IMAGE_SIZE[0] // 2 + 14, IMAGE_SIZE[1] // 2 - 10), 8, (230, 230, 230), -1)
            cv2.ellipse(canvas, (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2 + 16), (16, 7), 0, 0, 180, (210, 210, 210), 2)
            aug_u = apply_realistic_augmentation(canvas)
            cv2.imwrite(os.path.join(u_dir, f"{u_name}_{k:03d}.jpg"), aug_u)

    logger.info("Generated %d unknown impostor test samples across %d unknown identities", 75, len(unknown_profiles))

    # 4. Save class_names.json
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names_map, f, indent=2)
    logger.info("Saved class mapping to %s", CLASS_NAMES_PATH)

    # 5. Save person_metadata.json
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(person_metadata, f, indent=2)
    logger.info("Saved person metadata to %s", METADATA_PATH)

    return class_names_map, person_metadata


if __name__ == "__main__":
    print("=" * 65)
    print("Building Multi-Class Face Dataset and Validation Structure")
    print("=" * 65)
    c_map, meta = build_multiclass_dataset(target_samples_per_person=60)
    print(f"\nGenerated {len(c_map)} classes:")
    print(json.dumps(c_map, indent=2))
