"""
scripts/train_faces.py

GSSS Attendance Portal - Student Face Dataset Enrollment
Powered by InsightFace Antelopev2 (SCRFD 10G + Glint360k ResNet-100 ArcFace)

Features:
- Scans student photo folders directly from uploads/dataset/<USN>/
- Connects directly to MySQL `students` table by `student_number` (NO EXCEL DEPENDENCY)
- Extracts 512-dimensional normalized embeddings using InsightFace Antelopev2
- Stores individual photo embeddings in MySQL `face_embeddings` table
- Calculates student centroid embedding and updates `students.face_encoding`
- Avoids duplicate work by checking existing records unless --force is specified
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import Config
from backend.services.db_service import run_query
from backend.services.face_service import (
    get_face_app,
    normalize_embedding,
    get_students_map_from_db,
    ensure_student_in_db,
    EXPECTED_EMBEDDING_DIMENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train face embeddings directly from uploads/dataset folder using InsightFace Antelopev2.")
    parser.add_argument("--force", action="store_true", help="Force regenerate all embeddings even if already present in database.")
    args = parser.parse_args()

    dataset_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset")

    print("=" * 75)
    print(" GSSS ATTENDANCE PORTAL - INSIGHTFACE ANTELOPEV2 DATASET ENROLLMENT")
    print("=" * 75)
    print(f"Model          : InsightFace Antelopev2 (512-dimensional Glint360k)")
    print(f"Dataset Folder : {dataset_dir}")
    print(f"Force Retrain  : {args.force}")
    print("=" * 75)
    print()

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found at: {dataset_dir}")
        sys.exit(1)

    # 1. Load students directly from MySQL students table
    print("[1/3] Loading student records directly from MySQL database...")
    students_map = get_students_map_from_db()
    print(f"      Total students found in database: {len(students_map)}")
    print()

    # 2. Initialize Antelopev2 engine
    print("[2/3] Initializing InsightFace Antelopev2 engine (SCRFD + Glint360k)...")
    app = get_face_app()
    print("      Antelopev2 engine ready.")
    print()

    # 3. Process uploads/dataset/<USN>/
    print("[3/3] Scanning uploads/dataset/<USN>/ folders...")

    existing_rows = run_query(
        "SELECT student_id, image_path, embedding FROM face_embeddings WHERE embedding IS NOT NULL",
        fetch_all=True
    ) or []

    existing_cache = {}
    for r in existing_rows:
        sid = int(r["student_id"])
        ipath = (r.get("image_path") or "").replace("\\", "/").strip()
        if ipath and r.get("embedding"):
            existing_cache[(sid, ipath)] = r["embedding"]

    student_folders = [
        f.strip() for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    ]
    student_folders.sort()

    print(f"      Found {len(student_folders)} student folders in dataset.")
    print()
    print("Enrolling student face embeddings...")
    print("-" * 75)

    total_images_processed = 0
    total_embeddings_created = 0
    total_reused = 0
    students_with_faces = 0
    skipped_photos = 0

    for folder_name in student_folders:
        usn_clean = folder_name.strip().upper()
        sinfo = students_map.get(usn_clean)

        if not sinfo:
            sinfo = ensure_student_in_db(usn_clean)
            students_map[usn_clean] = sinfo

        student_id = sinfo["student_id"]
        student_name = sinfo.get("name", usn_clean)
        folder_path = os.path.join(dataset_dir, folder_name)

        # Collect image files
        image_files = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_files.append(os.path.join(folder_path, fname))

        if not image_files:
            print(f" [WARN] {usn_clean} | {student_name:<25} | No images found in folder.")
            continue

        student_embeddings = []
        new_for_student = 0
        reused_for_student = 0

        for img_path in image_files:
            total_images_processed += 1
            rel_path = os.path.relpath(img_path, Config.BASE_DIR).replace("\\", "/")
            cache_key = (student_id, rel_path)

            if not args.force and cache_key in existing_cache:
                raw_bytes = existing_cache[cache_key]
                try:
                    emb = np.frombuffer(raw_bytes, dtype=np.float32)
                    if emb.size == EXPECTED_EMBEDDING_DIMENSIONS:
                        student_embeddings.append(emb)
                        reused_for_student += 1
                        total_reused += 1
                        continue
                except Exception:
                    pass

            # Generate Antelopev2 embedding
            img = cv2.imread(img_path)
            if img is None:
                skipped_photos += 1
                continue

            faces = app.get(img)
            if not faces:
                skipped_photos += 1
                continue

            # Pick primary face with largest area
            primary_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            if primary_face.embedding is None:
                skipped_photos += 1
                continue

            emb = normalize_embedding(primary_face.embedding)
            emb_bytes = emb.astype(np.float32).tobytes()

            # Insert into face_embeddings table
            run_query(
                "INSERT INTO face_embeddings (student_id, image_path, embedding) VALUES (%s, %s, %s)",
                (student_id, rel_path, emb_bytes),
                commit=True
            )
            existing_cache[cache_key] = emb_bytes
            student_embeddings.append(emb)
            new_for_student += 1
            total_embeddings_created += 1

        if student_embeddings:
            centroid = np.mean(student_embeddings, axis=0)
            centroid = normalize_embedding(centroid)
            centroid_bytes = centroid.astype(np.float32).tobytes()
            run_query(
                "UPDATE students SET face_encoding = %s WHERE student_id = %s",
                (centroid_bytes, student_id),
                commit=True
            )
            students_with_faces += 1
            print(f" [OK] {usn_clean} | {student_name:<25} | {len(student_embeddings)} photos ({new_for_student} new, {reused_for_student} reused)")
        else:
            print(f" [FAIL] {usn_clean} | {student_name:<25} | No valid faces detected.")

    print("-" * 75)
    print()
    print("=" * 75)
    print(" INSIGHTFACE ANTELOPEV2 ENROLLMENT COMPLETE")
    print("=" * 75)
    print(f" Total Photos Scanned       : {total_images_processed}")
    print(f" New 512-d Embeddings Saved : {total_embeddings_created}")
    print(f" Reused Embeddings          : {total_reused}")
    print(f" Students Enrolled          : {students_with_faces} / {len(student_folders)}")
    print(f" Unusable Photos            : {skipped_photos}")
    print("=" * 75)


if __name__ == "__main__":
    main()