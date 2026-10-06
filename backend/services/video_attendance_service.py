"""
backend/services/video_attendance_service.py

Processes a classroom video:
    video -> faces -> FaceNet embeddings -> student matching -> attendance
"""

import os
import pickle
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace

from backend.services.db_service import run_query


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "retinaface"

# Cosine similarity threshold.
# Higher = stricter matching.
FACE_MATCH_THRESHOLD = 0.40

# Process every Nth frame.
# Increase this for faster processing of long videos.
FRAME_SKIP = 10


# ============================================================
# LOAD STORED EMBEDDINGS
# ============================================================

def load_known_embeddings():
    """
    Load all registered student face embeddings from MySQL.
    """

    rows = run_query(
        """
        SELECT
            embedding_id,
            student_id,
            embedding
        FROM face_embeddings
        """,
        fetch_all=True
    ) or []

    known_embeddings = []

    for row in rows:

        try:
            embedding = deserialize_embedding(
                row["embedding"]
            )

            if embedding is None:
                continue

            embedding = normalize_embedding(
                embedding
            )

            known_embeddings.append(
                {
                    "embedding_id": row["embedding_id"],
                    "student_id": row["student_id"],
                    "embedding": embedding
                }
            )

        except Exception as error:
            print(
                f"Could not load embedding "
                f"{row.get('embedding_id')}: {error}"
            )

    return known_embeddings


# ============================================================
# DESERIALIZE EMBEDDING
# ============================================================

def deserialize_embedding(value):
    """
    Convert the MySQL LONGBLOB back into a numpy array.

    Supports pickle and raw numpy byte formats.
    """

    if value is None:
        return None

    # MySQL may return bytearray.
    if isinstance(value, bytearray):
        value = bytes(value)

    # Pickled numpy array.
    if isinstance(value, bytes):

        try:
            result = pickle.loads(value)

            if isinstance(result, np.ndarray):
                return result.astype(
                    np.float32
                )

            if isinstance(result, (list, tuple)):
                return np.asarray(
                    result,
                    dtype=np.float32
                )

        except Exception:
            pass

        # Raw float32 bytes.
        if len(value) % 4 == 0:

            try:
                result = np.frombuffer(
                    value,
                    dtype=np.float32
                )

                if result.size > 0:
                    return result
            except Exception:
                pass

        # Raw float64 bytes.
        if len(value) % 8 == 0:

            try:
                result = np.frombuffer(
                    value,
                    dtype=np.float64
                ).astype(np.float32)

                if result.size > 0:
                    return result
            except Exception:
                pass

    # Already a numpy array.
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)

    # List/tuple.
    if isinstance(value, (list, tuple)):
        return np.asarray(
            value,
            dtype=np.float32
        )

    return None


# ============================================================
# NORMALIZE
# ============================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(
            "Embedding has zero magnitude."
        )

    return embedding / norm


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = normalize_embedding(a)
    b = normalize_embedding(b)

    return float(
        np.dot(a, b)
    )


# ============================================================
# FIND BEST STUDENT
# ============================================================

def find_best_student(
    face_embedding,
    known_embeddings
):
    """
    Compare one detected face against all registered
    student embeddings.
    """

    best_student_id = None
    best_score = -1.0

    for item in known_embeddings:

        stored_embedding = item["embedding"]

        # Different dimensions cannot be compared.
        if (
            face_embedding.shape
            != stored_embedding.shape
        ):
            continue

        score = cosine_similarity(
            face_embedding,
            stored_embedding
        )

        if score > best_score:

            best_score = score
            best_student_id = item["student_id"]

    if (
        best_student_id is not None
        and best_score >= FACE_MATCH_THRESHOLD
    ):
        return (
            best_student_id,
            best_score
        )

    return (
        None,
        best_score
    )


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(student_id):
    """
    Mark a student Present once for the current day.

    Returns:
        True  -> new attendance inserted
        False -> already marked today
    """

    existing = run_query(
        """
        SELECT attendance_id
        FROM attendance
        WHERE student_id = %s
          AND DATE(check_in_time) = CURDATE()
        LIMIT 1
        """,
        (student_id,),
        fetch_one=True
    )

    if existing:
        return False

    run_query(
        """
        INSERT INTO attendance
        (
            student_id,
            check_in_time,
            status
        )
        VALUES
        (
            %s,
            CURRENT_TIMESTAMP,
            'Present'
        )
        """,
        (student_id,),
        commit=True
    )

    return True


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):
    """
    Process classroom video and mark recognized students.

    Returns:
        {
            "face_count": number of detected faces,
            "attendance_count": number of new attendance records,
            "students": recognized student IDs
        }
    """

    if not os.path.exists(video_path):
        raise ValueError(
            f"Video does not exist: {video_path}"
        )

    known_embeddings = load_known_embeddings()

    if not known_embeddings:
        raise ValueError(
            "No face embeddings are registered."
        )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():
        raise ValueError(
            "Unable to open classroom video."
        )

    frame_number = 0

    detected_face_count = 0

    recognized_students = set()

    attendance_count = 0

    try:

        while True:

            success, frame = video.read()

            if not success:
                break

            frame_number += 1

            # Process every Nth frame.
            if frame_number % FRAME_SKIP != 0:
                continue

            # DeepFace can receive the OpenCV frame directly.
            results = DeepFace.represent(
                img_path=frame,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
                normalization="Facenet"
            )

            if not results:
                continue

            detected_face_count += len(results)

            for result in results:

                # DeepFace may return a result without
                # a usable embedding.
                if "embedding" not in result:
                    continue

                face_embedding = np.asarray(
                    result["embedding"],
                    dtype=np.float32
                )

                try:
                    face_embedding = normalize_embedding(
                        face_embedding
                    )
                except ValueError:
                    continue

                student_id, score = find_best_student(
                    face_embedding,
                    known_embeddings
                )

                if student_id is None:
                    continue

                # Prevent the same student from being
                # counted repeatedly in the same video.
                if student_id in recognized_students:
                    continue

                recognized_students.add(
                    student_id
                )

                if mark_attendance(
                    student_id
                ):
                    attendance_count += 1

    finally:

        video.release()

    return {
        "face_count": detected_face_count,
        "attendance_count": attendance_count,
        "students": list(
            recognized_students
        )
    }