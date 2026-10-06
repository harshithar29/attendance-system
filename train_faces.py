"""
train_faces.py

Standalone CLI utility to delete existing trained embeddings and retrain fresh
from the reference photos in uploads/dataset/<USN>/ using InsightFace Antelopev2 (512-d).

Usage:
    python train_faces.py
    python train_faces.py --clean
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.services.db_service import run_query
from backend.services.face_service import (
    get_face_app,
    sync_student_embeddings
)


def clear_existing_embeddings():
    """Wipes all previously trained face embeddings from the database."""
    print("=" * 70)
    print("1. CLEARING PREVIOUSLY TRAINED EMBEDDINGS...")
    print("=" * 70)

    try:
        # Delete from face_embeddings
        del_count = run_query("DELETE FROM face_embeddings", commit=True)
        print("   -> Cleared face_embeddings table.")
    except Exception as e:
        print(f"   -> Warning while clearing face_embeddings: {e}")

    try:
        # Reset face_encoding centroids in students table
        run_query("UPDATE students SET face_encoding = NULL", commit=True)
        print("   -> Reset students.face_encoding centroids to NULL.")
    except Exception as e:
        print(f"   -> Warning while resetting students table: {e}")

    print("   -> Database embeddings successfully wiped clean.\n")


def run_training(dataset_dir=None, clean=True):
    """Retrains all student face embeddings from uploads/dataset/."""
    if dataset_dir is None:
        dataset_dir = os.path.join(BASE_DIR, "uploads", "dataset")

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found at: {dataset_dir}")
        sys.exit(1)

    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    print("=" * 70)
    print("2. FOUND STUDENT DATASET FOLDERS")
    print("=" * 70)
    print(f"   Dataset Location: {dataset_dir}")
    print(f"   Total Student Folders: {len(folders)}")
    print("=" * 70 + "\n")

    if clean:
        clear_existing_embeddings()

    print("=" * 70)
    print("3. STARTING FRESH ANTELOPEV2 (512-D) TRAINING...")
    print("=" * 70)

    start_time = time.time()

    # Pre-initialize face app
    print("   -> Initializing InsightFace Antelopev2 (SCRFD 10G + Glint360k)...")
    get_face_app()
    print("   -> Model initialized successfully.")
    print("   -> Processing photos student-by-student...\n")

    report = sync_student_embeddings(
        dataset_dir=dataset_dir,
        force_refresh=True
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"   Elapsed Time:              {elapsed:.2f} seconds")
    print(f"   Total Students in Dataset: {report.get('total_students', 0)}")
    print(f"   Students with Photos:      {len(report.get('students_with_photos', []))}")
    print(f"   New Embeddings Generated:  {report.get('new_embeddings_generated', 0)}")
    print(f"   Total Active Embeddings:   {report.get('total_active_embeddings', 0)}")

    missing = report.get("students_missing_photos", [])
    if missing:
        print(f"   Students Missing Photos:   {len(missing)}")
        for m in missing[:5]:
            print(f"      - {m.get('student_number')}: {m.get('name')}")
        if len(missing) > 5:
            print(f"      ... and {len(missing) - 5} more.")

    print("=" * 70)
    print("The system is now fully trained and ready for fresh video processing.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain student face dataset using InsightFace Antelopev2.")
    parser.add_argument("--clean", action="store_true", default=True, help="Wipe existing embeddings before training (default: True)")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Keep existing embeddings and only train missing photos")
    parser.add_argument("--dataset-dir", default=None, help="Custom dataset path (default: uploads/dataset)")

    args = parser.parse_args()
    run_training(dataset_dir=args.dataset_dir, clean=args.clean)
