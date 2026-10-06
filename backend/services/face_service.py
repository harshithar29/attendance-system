"""
backend/services/face_service.py

Student face registration, dataset loading, and InsightFace Antelopev2 embedding generation.
- Model: InsightFace Antelopev2 (SCRFD 10G detector + Glint360k ResNet-100 ArcFace recognizer)
- Embedding dimensions: 512 float32 (L2 normalized)
- Dataset source: uploads/dataset/<USN>/
- Database: Maps directly to MySQL `students` table by `student_number` (No Excel dependency).
"""

import os
import logging
import hashlib

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config.config import Config
from backend.services.db_service import run_query


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "antelopev2"
EXPECTED_EMBEDDING_DIMENSIONS = 512

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".webp",
    ".bmp"
)

# Global singleton for InsightFace Antelopev2
_face_app = None


def get_face_app():
    """
    Initialize and return the InsightFace Antelopev2 FaceAnalysis instance (singleton).
    Configured for CPU execution provider.
    """
    global _face_app
    if _face_app is None:
        logger.info("Initializing InsightFace Antelopev2 FaceAnalysis engine...")
        try:
            app = FaceAnalysis(
                name=MODEL_NAME,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"]
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = app
            logger.info("InsightFace Antelopev2 loaded successfully (512-d Glint360k / SCRFD).")
        except Exception as e:
            logger.exception("Failed to initialize InsightFace Antelopev2: %s", e)
            raise
    return _face_app


# ============================================================
# EMBEDDING NORMALIZATION
# ============================================================

def normalize_embedding(embedding):
    """
    Convert embedding to normalized float32 512-d array.
    """
    if embedding is None:
        raise ValueError("Embedding is None.")

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size != EXPECTED_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIMENSIONS} dimensions, received {embedding.size}."
        )

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has invalid magnitude.")

    return (embedding / norm).astype(np.float32)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    if not image_path:
        raise ValueError("Image path is empty.")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


# ============================================================
# GENERATE EMBEDDING (INSIGHTFACE ANTELOPEV2)
# ============================================================

def generate_embedding(image_path):
    """
    Generate one normalized 512-d InsightFace Antelopev2 embedding for a face image.
    If multiple faces are detected, selects the primary (largest area) face.
    """
    image = load_image(image_path)
    height, width = image.shape[:2]

    if width < 30 or height < 30:
        raise ValueError(f"Image resolution is too small ({width}x{height}).")

    app = get_face_app()
    faces = app.get(image)

    if not faces:
        # Fallback: try minor upscale if small image
        if max(width, height) < 320:
            scale = 320.0 / max(width, height)
            upscaled = cv2.resize(image, (int(width * scale), int(height * scale)))
            faces = app.get(upscaled)

    if not faces:
        raise ValueError(f"No face detected in {os.path.basename(image_path)}.")

    # Select the face with largest bounding box area
    target_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    if target_face.embedding is None:
        raise ValueError("Face detected but Antelopev2 embedding was not generated.")

    return normalize_embedding(target_face.embedding)


# ============================================================
# VALIDATE STUDENT IMAGE
# ============================================================

def validate_student_image(image_path):
    try:
        embedding = generate_embedding(image_path)
        return {
            "valid": True,
            "message": "Valid student face image (Antelopev2 512-d).",
            "embedding_dimensions": int(embedding.size)
        }
    except Exception as error:
        return {
            "valid": False,
            "message": str(error),
            "embedding_dimensions": None
        }


# ============================================================
# DATASET PHOTO SOURCE (uploads/dataset/<USN>/)
# ============================================================

def get_student_reference_photos(usn, dataset_dir=None):
    """
    Collects reference photos for a given student USN exclusively from:
      uploads/dataset/{usn}/
    Deduplicates identical files using MD5 hashes.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset")

    photos_found = []
    usn_clean = str(usn).strip().upper()

    target_folder = None
    if os.path.isdir(dataset_dir):
        for entry in os.listdir(dataset_dir):
            if entry.strip().upper() == usn_clean:
                target_folder = os.path.join(dataset_dir, entry)
                break

    if target_folder and os.path.isdir(target_folder):
        for fname in os.listdir(target_folder):
            if fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                photos_found.append(os.path.join(target_folder, fname))

    # Deduplicate by file content MD5 hash
    unique_photos = []
    seen_hashes = set()
    for p in photos_found:
        try:
            with open(p, "rb") as fp:
                file_hash = hashlib.md5(fp.read()).hexdigest()
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_photos.append(p)
        except Exception:
            pass

    return unique_photos


# ============================================================
# DIRECT DATABASE STUDENT MAPPER (NO EXCEL)
# ============================================================

def get_students_map_from_db():
    """
    Loads all student records directly from MySQL `students` table.
    Returns dict: { USN: { student_id, name, student_number, email } }
    """
    rows = run_query(
        "SELECT student_id, name, student_number, email FROM students",
        fetch_all=True
    ) or []

    students_map = {}
    for row in rows:
        usn = (row.get("student_number") or "").strip().upper()
        if usn:
            students_map[usn] = {
                "student_id": int(row["student_id"]),
                "name": row.get("name") or usn,
                "student_number": usn,
                "email": row.get("email") or f"{usn.lower()}@student.local"
            }
    return students_map


def ensure_student_in_db(usn, name=None):
    """
    Ensures a student USN exists in the MySQL students table.
    Returns the student dictionary.
    """
    usn_clean = str(usn).strip().upper()
    existing = run_query(
        "SELECT student_id, name, student_number, email FROM students WHERE UPPER(student_number) = %s LIMIT 1",
        (usn_clean,),
        fetch_one=True
    )
    if existing:
        return {
            "student_id": int(existing["student_id"]),
            "name": existing.get("name") or usn_clean,
            "student_number": usn_clean,
            "email": existing.get("email") or f"{usn_clean.lower()}@student.local"
        }

    display_name = name.strip() if name else usn_clean
    email = f"{usn_clean.lower()}@student.local"
    run_query(
        "INSERT INTO students (name, email, student_number) VALUES (%s, %s, %s)",
        (display_name, email, usn_clean),
        commit=True
    )
    new_row = run_query(
        "SELECT student_id, name, student_number, email FROM students WHERE UPPER(student_number) = %s LIMIT 1",
        (usn_clean,),
        fetch_one=True
    )
    return {
        "student_id": int(new_row["student_id"]),
        "name": new_row.get("name") or display_name,
        "student_number": usn_clean,
        "email": email
    }


# ============================================================
# SYNC STUDENT EMBEDDINGS FROM uploads/dataset/ (NO EXCEL)
# ============================================================

def sync_student_embeddings(excel_path=None, dataset_dir=None, force_refresh=False):
    """
    Synchronizes student reference photos and InsightFace Antelopev2 embeddings.
    - Connects directly to uploads/dataset/<USN>/ and MySQL `students` table.
    - DOES NOT require or connect studentdb.xlsx.
    - Generates 512-d normalized Antelopev2 embeddings.
    - Reuses existing valid 512-d embeddings from MySQL face_embeddings table.
    - Computes each student's normalized centroid and stores in students.face_encoding.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset")

    if not os.path.isdir(dataset_dir):
        logger.warning("Dataset directory not found at: %s", dataset_dir)
        return {
            "total_students": 0,
            "students_with_photos": [],
            "students_missing_photos": [],
            "new_embeddings_generated": 0,
            "reused_embeddings": 0,
            "total_active_embeddings": 0
        }

    # 1. Fetch student folders in uploads/dataset/
    student_folders = [
        f.strip().upper() for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    ]
    student_folders.sort()

    # 2. Get students map directly from MySQL database
    db_students = get_students_map_from_db()

    # 3. Fetch all existing stored embeddings to reuse
    existing_rows = run_query(
        "SELECT student_id, image_path, embedding FROM face_embeddings WHERE embedding IS NOT NULL",
        fetch_all=True
    ) or []

    existing_map = {}
    for r in existing_rows:
        sid = int(r["student_id"])
        ipath = (r.get("image_path") or "").replace("\\", "/").strip()
        if ipath and r.get("embedding"):
            existing_map[(sid, ipath)] = r["embedding"]

    new_embeddings_generated = 0
    reused_embeddings = 0
    students_with_photos = []
    students_missing_photos = []

    for usn in student_folders:
        sinfo = db_students.get(usn)
        if not sinfo:
            sinfo = ensure_student_in_db(usn)
            db_students[usn] = sinfo

        student_id = sinfo["student_id"]
        student_name = sinfo["name"]

        # Collect photos exclusively from uploads/dataset/<USN>/
        photos = get_student_reference_photos(
            usn=usn,
            dataset_dir=dataset_dir
        )

        if not photos:
            students_missing_photos.append({
                "student_id": student_id,
                "student_number": usn,
                "name": student_name,
                "photo_count": 0
            })
            continue

        valid_student_embeddings = []
        for img_path in photos:
            rel_path = os.path.relpath(img_path, Config.BASE_DIR).replace("\\", "/")
            key = (student_id, rel_path)

            if not force_refresh and key in existing_map:
                # REUSE existing embedding if size matches 512 dimensions (2048 bytes)
                raw_bytes = existing_map[key]
                try:
                    emb = np.frombuffer(raw_bytes, dtype=np.float32)
                    if emb.size == EXPECTED_EMBEDDING_DIMENSIONS:
                        valid_student_embeddings.append(emb)
                        reused_embeddings += 1
                        continue
                except Exception:
                    pass

            # Generate new 512-d Antelopev2 embedding
            try:
                emb = generate_embedding(img_path)
                valid_student_embeddings.append(emb)
                emb_bytes = emb.astype(np.float32).tobytes()

                # Insert into face_embeddings
                run_query(
                    "INSERT INTO face_embeddings (student_id, image_path, embedding) VALUES (%s, %s, %s)",
                    (student_id, rel_path, emb_bytes),
                    commit=True
                )
                existing_map[key] = emb_bytes
                new_embeddings_generated += 1
                logger.info("Generated Antelopev2 512-d embedding for %s (%s): %s", usn, student_name, os.path.basename(img_path))
            except Exception as err:
                logger.warning("Could not generate Antelopev2 embedding for %s (%s): %s", os.path.basename(img_path), usn, err)

        if valid_student_embeddings:
            centroid = np.mean(valid_student_embeddings, axis=0)
            centroid = normalize_embedding(centroid)
            centroid_bytes = centroid.astype(np.float32).tobytes()
            run_query(
                "UPDATE students SET face_encoding = %s WHERE student_id = %s",
                (centroid_bytes, student_id),
                commit=True
            )
            students_with_photos.append({
                "student_id": student_id,
                "student_number": usn,
                "name": student_name,
                "photo_count": len(valid_student_embeddings)
            })
        else:
            students_missing_photos.append({
                "student_id": student_id,
                "student_number": usn,
                "name": student_name,
                "photo_count": 0
            })

    report = {
        "total_students": len(student_folders),
        "students_with_photos": students_with_photos,
        "students_missing_photos": students_missing_photos,
        "new_embeddings_generated": new_embeddings_generated,
        "reused_embeddings": reused_embeddings,
        "total_active_embeddings": reused_embeddings + new_embeddings_generated
    }
    logger.info(
        "Antelopev2 student sync complete: %s students with photos (%s active embeddings, %s new, %s reused), %s missing photos.",
        len(students_with_photos),
        report["total_active_embeddings"],
        new_embeddings_generated,
        reused_embeddings,
        len(students_missing_photos)
    )
    return report


def train_student_face_dataset(dataset_dir=None, force_retrain=False, **kwargs):
    """
    Main dataset training function.
    Directly connects uploads/dataset/<USN>/ with MySQL students table.
    """
    res = sync_student_embeddings(
        dataset_dir=dataset_dir,
        force_refresh=force_retrain
    )
    return {
        "total_students": res["total_students"],
        "trained_students": len(res["students_with_photos"]),
        "total_embeddings": res["total_active_embeddings"],
        "skipped_students": len(res["students_missing_photos"]),
        "new_embeddings_generated": res["new_embeddings_generated"],
        "reused_embeddings": res["reused_embeddings"]
    }