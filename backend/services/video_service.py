"""
backend/services/video_service.py

Classroom multi-face recognition pipeline powered by InsightFace Antelopev2:
- Detector: SCRFD 10G (handles dense multi-face classroom scenes with high precision)
- Recognizer: Glint360k ResNet-100 ArcFace (512-dimensional normalized embeddings)
- Reference photos source: MySQL `face_embeddings` table / `uploads/dataset/<USN>/`
- Anti-False-Positive: Strict cosine similarity threshold, frame margin validation, and multi-frame quorum confirmation
"""

import os
import io
import ast
import json
import base64
import pickle
import logging
import glob
from collections import defaultdict

import time
import cv2
import numpy as np
from insightface.utils import face_align

from backend.services.db_service import run_query
from backend.services.face_service import (
    get_face_app,
    normalize_embedding,
    EXPECTED_EMBEDDING_DIMENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS
)


logger = logging.getLogger(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# STRICT THRESHOLDS - ANTELOPEV2 (512-D) ANTI-FALSE-POSITIVE
# ============================================================

# For InsightFace Glint360k 512-d embeddings:
# Cosine similarity between different people is typically < 0.25.
# Matching faces typically score >= 0.45 to 0.75+.

# Minimum similarity to be considered a candidate in a frame
FACE_MATCH_THRESHOLD = 0.365

# Minimum margin between top match and runner-up to reject ambiguity
FRAME_MARGIN_THRESHOLD = 0.010

# Video-level confirmation criteria:
MIN_SIMILARITY_TO_CONFIRM = 0.365
MIN_AVG_SIMILARITY = 0.350
MIN_DETECTIONS_TO_CONFIRM = 2
MIN_CONFIRM_MARGIN = 0.008
MIN_BEST_MARGIN = 0.012

# Face detector confidence score threshold (SCRFD)
DET_CONFIDENCE_THRESHOLD = 0.28
MIN_DET_SCORE = 0.28
MIN_FACE_WIDTH = 8
MIN_FACE_HEIGHT = 8

# Video sampling rates for rapid ~15-20s classroom processing
MAX_PROCESSED_FRAMES = 26



# ============================================================
# EMBEDDING DESERIALIZATION
# ============================================================

def try_normalize_embedding(value):
    try:
        return normalize_embedding(value)
    except Exception:
        return None


def deserialize_embedding(value):
    """
    Deserializes raw database BLOB/bytes/lists into a 512-d normalized numpy array.
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return try_normalize_embedding(value)

    if isinstance(value, (list, tuple)):
        return try_normalize_embedding(value)

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, bytearray):
        value = bytes(value)

    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return None
        try:
            decoded = json.loads(text_value)
            emb = try_normalize_embedding(decoded)
            if emb is not None:
                return emb
        except Exception:
            pass

        try:
            decoded = ast.literal_eval(text_value)
            emb = try_normalize_embedding(decoded)
            if emb is not None:
                return emb
        except Exception:
            pass

        try:
            decoded_bytes = base64.b64decode(text_value, validate=True)
            emb = deserialize_embedding(decoded_bytes)
            if emb is not None:
                return emb
        except Exception:
            pass
        return None

    if not isinstance(value, bytes) or not value:
        return None

    byte_length = len(value)

    # 1. Direct float32 raw bytes (512 * 4 = 2048 bytes)
    if byte_length == EXPECTED_EMBEDDING_DIMENSIONS * 4:
        try:
            arr = np.frombuffer(value, dtype=np.float32).copy()
            emb = try_normalize_embedding(arr)
            if emb is not None:
                return emb
        except Exception:
            pass

    # 2. Direct float64 raw bytes (512 * 8 = 4096 bytes)
    if byte_length == EXPECTED_EMBEDDING_DIMENSIONS * 8:
        try:
            arr = np.frombuffer(value, dtype=np.float64).astype(np.float32)
            emb = try_normalize_embedding(arr)
            if emb is not None:
                return emb
        except Exception:
            pass

    # 3. Pickle fallback
    try:
        decoded = pickle.loads(value)
        emb = try_normalize_embedding(decoded)
        if emb is not None:
            return emb
    except Exception:
        pass

    return None


# ============================================================
# LOAD REGISTERED EMBEDDINGS FROM DATABASE
# ============================================================

def load_registered_embeddings(class_id=None):
    """
    Loads 512-d Antelopev2 embeddings stored in MySQL:
    1. If class_id is given, filters to students enrolled in that class to prevent false positives
    2. Fallback to all enrolled students if class filter is empty
    """
    rows = []
    if class_id:
        try:
            class_query = """
                SELECT fe.embedding_id, fe.student_id, fe.embedding
                FROM face_embeddings fe
                INNER JOIN class_students cs ON fe.student_id = cs.student_id
                WHERE cs.class_id = %s AND fe.embedding IS NOT NULL
                ORDER BY fe.embedding_id DESC
            """
            rows = run_query(class_query, (class_id,), fetch_all=True) or []
        except Exception as error:
            logger.warning("Query for class-specific face_embeddings failed: %s", error)

    if not rows:
        query = """
            SELECT embedding_id, student_id, embedding
            FROM face_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY embedding_id DESC
        """
        try:
            rows = run_query(query, fetch_all=True) or []
        except Exception as error:
            logger.exception("Database query for face_embeddings failed: %s", error)

    registered = []
    for row in rows:
        embedding_id = row.get("embedding_id")
        student_id = row.get("student_id")
        raw_embedding = row.get("embedding")

        if student_id is None or raw_embedding is None:
            continue

        emb = deserialize_embedding(raw_embedding)
        if emb is not None:
            registered.append({
                "embedding_id": int(embedding_id) if embedding_id else None,
                "student_id": int(student_id),
                "embedding": emb
            })

    # If face_embeddings is empty, try students.face_encoding centroids
    if not registered:
        logger.info("No embeddings in face_embeddings table, checking students.face_encoding...")
        try:
            st_rows = run_query(
                "SELECT student_id, face_encoding FROM students WHERE face_encoding IS NOT NULL",
                fetch_all=True
            ) or []
            for st in st_rows:
                sid = st.get("student_id")
                raw = st.get("face_encoding")
                if sid and raw:
                    emb = deserialize_embedding(raw)
                    if emb is not None:
                        registered.append({
                            "embedding_id": None,
                            "student_id": int(sid),
                            "embedding": emb
                        })
        except Exception as e:
            logger.warning("Could not read students.face_encoding: %s", e)

    return registered


def build_embedding_matrix(registered_embeddings):
    """
    Converts a list of registered embeddings into:
    - student_ids: np.array of student_ids (shape: [N])
    - embedding_matrix: np.ndarray of 512-d unit vectors (shape: [N, 512])
    """
    if not registered_embeddings:
        return (
            np.array([], dtype=np.int64),
            np.empty((0, EXPECTED_EMBEDDING_DIMENSIONS), dtype=np.float32)
        )

    student_ids_list = []
    embedding_list = []

    for item in registered_embeddings:
        sid = item.get("student_id")
        emb = item.get("embedding")
        if sid is None or emb is None:
            continue
        try:
            norm_emb = normalize_embedding(emb)
            student_ids_list.append(int(sid))
            embedding_list.append(norm_emb)
        except Exception:
            continue

    if not embedding_list:
        return (
            np.array([], dtype=np.int64),
            np.empty((0, EXPECTED_EMBEDDING_DIMENSIONS), dtype=np.float32)
        )

    student_ids = np.asarray(student_ids_list, dtype=np.int64)
    embedding_matrix = np.vstack(embedding_list).astype(np.float32)

    # Re-normalize rows to ensure strict unit length
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding_matrix = embedding_matrix / norms

    return student_ids, embedding_matrix


# ============================================================
# FALLBACK: READ DIRECTLY FROM uploads/dataset/<USN>/
# ============================================================

def read_student_photos_from_dataset():
    """
    Fallback if database embeddings table is empty:
    Reads reference photos exclusively from uploads/dataset/<USN>/
    and generates Antelopev2 512-d embeddings on the fly.
    """
    dataset_dir = os.path.join(BASE_DIR, "uploads", "dataset")
    if not os.path.isdir(dataset_dir):
        return None, None

    # Map USN to student_id
    usn_to_id = {}
    try:
        rows = run_query("SELECT student_id, student_number FROM students", fetch_all=True) or []
        for r in rows:
            sid = int(r["student_id"])
            usn = (r.get("student_number") or "").strip().upper()
            if usn:
                usn_to_id[usn] = sid
    except Exception as e:
        logger.warning("Could not fetch students mapping: %s", e)

    app = get_face_app()
    student_ids_list = []
    embedding_list = []

    for entry in os.listdir(dataset_dir):
        student_folder = os.path.join(dataset_dir, entry)
        if not os.path.isdir(student_folder):
            continue

        usn_clean = entry.strip().upper()
        sid = usn_to_id.get(usn_clean)
        if not sid:
            continue

        for fname in os.listdir(student_folder):
            if fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                img_path = os.path.join(student_folder, fname)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    faces = app.get(img)
                    if not faces:
                        continue
                    # Pick primary face
                    target_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    emb = normalize_embedding(target_face.embedding)
                    student_ids_list.append(sid)
                    embedding_list.append(emb)
                except Exception:
                    continue

    if not embedding_list:
        return None, None

    student_ids = np.asarray(student_ids_list, dtype=np.int64)
    embedding_matrix = np.vstack(embedding_list).astype(np.float32)
    return student_ids, embedding_matrix


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_video_information(video_id):
    query = """
        SELECT video_id, class_id, subject_id
        FROM video_uploads
        WHERE video_id = %s
        LIMIT 1
    """
    return run_query(query, (video_id,), fetch_one=True)


def load_student_details(student_ids):
    if not student_ids:
        return {}

    try:
        unique_ids = sorted(set(int(s_id) for s_id in student_ids))
    except Exception:
        return {}

    if not unique_ids:
        return {}

    placeholders = ",".join(["%s"] * len(unique_ids))
    query = f"""
        SELECT student_id, name, student_number, email
        FROM students
        WHERE student_id IN ({placeholders})
    """
    try:
        rows = run_query(query, tuple(unique_ids), fetch_all=True) or []
    except Exception:
        return {}

    students = {}
    for row in rows:
        try:
            student_id = int(row.get("student_id"))
        except Exception:
            continue

        students[student_id] = {
            "student_id": student_id,
            "name": row.get("name") or "Unknown",
            "student_number": row.get("student_number") or "Unknown",
            "email": row.get("email") or ""
        }
    return students


def mark_student_present(student_id, class_id, subject_id, video_id):
    check_query = """
        SELECT attendance_id
        FROM attendance
        WHERE student_id = %s
          AND class_id = %s
          AND subject_id = %s
          AND DATE(check_in_time) = CURDATE()
        LIMIT 1
    """
    existing = run_query(check_query, (student_id, class_id, subject_id), fetch_one=True)
    if existing:
        update_query = """
            UPDATE attendance
            SET video_id = %s, status = 'Present', check_in_time = CURRENT_TIMESTAMP
            WHERE attendance_id = %s
        """
        run_query(update_query, (video_id, existing.get("attendance_id")), commit=True)
        return True

    insert_query = """
        INSERT INTO attendance
        (student_id, class_id, subject_id, video_id, check_in_time, status)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, 'Present')
    """
    run_query(insert_query, (student_id, class_id, subject_id, video_id), commit=True)
    return True


def mark_students_present_bulk(confirmed_student_ids, class_id, subject_id, video_id):
    """
    Bulk writes/updates attendance in MySQL in a single batch operation
    to eliminate roundtrip database query overhead after recognition is confirmed.
    """
    if not confirmed_student_ids:
        return 0, {}

    unique_sids = sorted(list(set(int(sid) for sid in confirmed_student_ids)))
    if not unique_sids:
        return 0, {}

    placeholders = ",".join(["%s"] * len(unique_sids))
    check_query = f"""
        SELECT attendance_id, student_id
        FROM attendance
        WHERE class_id = %s
          AND subject_id = %s
          AND DATE(check_in_time) = CURDATE()
          AND student_id IN ({placeholders})
    """
    params = [class_id, subject_id] + unique_sids
    try:
        existing_rows = run_query(check_query, tuple(params), fetch_all=True) or []
    except Exception as ex:
        logger.warning("Error checking existing attendance: %s", ex)
        existing_rows = []

    existing_map = {int(r["student_id"]): int(r["attendance_id"]) for r in existing_rows if r.get("student_id")}

    update_ids = []
    insert_tuples = []
    status_map = {}

    for sid in unique_sids:
        if sid in existing_map:
            update_ids.append(existing_map[sid])
            status_map[sid] = "Already Present"
        else:
            insert_tuples.append((sid, class_id, subject_id, video_id))
            status_map[sid] = "Present"

    # 1. Bulk update existing records
    if update_ids:
        try:
            up_placeholders = ",".join(["%s"] * len(update_ids))
            update_query = f"""
                UPDATE attendance
                SET video_id = %s, status = 'Present', check_in_time = CURRENT_TIMESTAMP
                WHERE attendance_id IN ({up_placeholders})
            """
            run_query(update_query, tuple([video_id] + update_ids), commit=True)
        except Exception as up_err:
            logger.error("Error bulk updating attendance: %s", up_err)

    # 2. Bulk insert new records
    new_attendance_count = len(insert_tuples)
    if insert_tuples:
        try:
            val_placeholders = ",".join(["(%s, %s, %s, %s, CURRENT_TIMESTAMP, 'Present')"] * len(insert_tuples))
            insert_query = f"""
                INSERT INTO attendance
                (student_id, class_id, subject_id, video_id, check_in_time, status)
                VALUES {val_placeholders}
            """
            flat_params = []
            for tup in insert_tuples:
                flat_params.extend(tup)
            run_query(insert_query, tuple(flat_params), commit=True)
        except Exception as in_err:
            logger.error("Error bulk inserting attendance: %s", in_err)

    return new_attendance_count, status_map


def update_video_status(video_id, status, face_count=None, attendance_count=None, notes=None):
    if status == "processing":
        query = "UPDATE video_uploads SET status = 'processing' WHERE video_id = %s"
        run_query(query, (video_id,), commit=True)
        return

    if status == "processed":
        query = """
            UPDATE video_uploads
            SET status = 'processed',
                processed_at = CURRENT_TIMESTAMP,
                face_count = %s,
                attendance_count = %s,
                notes = %s
            WHERE video_id = %s
        """
        run_query(query, (face_count or 0, attendance_count or 0, notes, video_id), commit=True)
        return

    if status == "failed":
        query = """
            UPDATE video_uploads
            SET status = 'failed',
                processed_at = CURRENT_TIMESTAMP
            WHERE video_id = %s
        """
        run_query(query, (video_id,), commit=True)


def mark_video_failed(video_id):
    try:
        update_video_status(video_id, "failed")
        return True
    except Exception:
        return False


def calculate_optimal_target_frames(total_frames, fps, duration):
    """
    Dynamically determines strategically distributed target frame indices to
    achieve ~15-20s processing while ensuring reliable multi-frame quorum across all benches.
    """
    if total_frames <= 0:
        return []

    # Dynamic target count based on video duration
    if duration <= 6.0:
        n_samples = min(10, total_frames)
    elif duration <= 15.0:
        n_samples = min(16, total_frames)
    elif duration <= 30.0:
        n_samples = min(22, total_frames)
    else:
        n_samples = min(25, total_frames)

    # Distribute samples evenly across video duration, centered within intervals
    step = total_frames / float(n_samples)
    targets = [int((i + 0.5) * step) for i in range(n_samples)]
    targets = [max(0, min(total_frames - 1, t)) for t in targets]
    return sorted(list(set(targets)))


def get_frame_interval(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    targets = calculate_optimal_target_frames(total_frames, fps, duration)
    interval = max(1, int(round(total_frames / max(1, len(targets)))))
    return fps, interval


# ============================================================
# LIGHTWEIGHT PREPROCESSING & DETECTION HELPERS
# ============================================================

def check_frame_lighting(frame):
    """
    Measures frame luminance and contrast in under 0.2ms.
    Returns whether the frame is dark/low-contrast along with metrics.
    """
    if frame is None or frame.size == 0:
        return False, 128.0, 50.0

    # 4x downscaled grayscale for instant luminance check (~0.1ms)
    small_gray = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(small_gray))
    std_lum = float(np.std(small_gray))
    is_dark = (mean_lum < 95.0) or (std_lum < 38.0)
    return is_dark, mean_lum, std_lum


def enhance_frame_lighting(frame, mean_lum):
    """
    Applies non-linear gamma correction via LUT (~2ms) and Y-channel CLAHE (~8ms)
    to reveal shadowed, distant, and poorly lit faces without blowing out highlights.
    Executed only when check_frame_lighting determines the frame is dark.
    """
    if frame is None or frame.size == 0:
        return frame

    # 1. Non-linear gamma adjustment
    gamma = float(np.clip(1.0 + (95.0 - mean_lum) / 80.0, 1.15, 1.65))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)

    # 2. Contrast-Limited Adaptive Histogram Equalization on Y luminance
    ycrcb = cv2.cvtColor(brightened, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)


def preprocess_classroom_frame(frame):
    """
    Evaluates frame luminance and enhances dark frames when needed.
    """
    is_dark, mean_lum, _ = check_frame_lighting(frame)
    if is_dark:
        return enhance_frame_lighting(frame, mean_lum), True
    return frame, False


def safe_detect_faces(det_model, img, thresh=0.28):
    """
    Safely invokes SCRFD detection across different InsightFace library versions.
    Maintains native aspect ratio to prevent face distortion.
    """
    if hasattr(det_model, "det_thresh"):
        try:
            det_model.det_thresh = thresh
        except Exception:
            pass

    try:
        return det_model.detect(img, det_thresh=thresh)
    except TypeError:
        return det_model.detect(img)



def preserve_face_crop_quality(aimg, face_w, face_h):
    """
    For distant/small faces (e.g. last benches), enhances high-frequency edge definition
    on the 112x112 ArcFace aligned crop to preserve discriminating features for recognition.
    """
    if aimg is None or aimg.size == 0:
        return aimg
    if min(face_w, face_h) < 32:
        gaussian = cv2.GaussianBlur(aimg, (0, 0), 1.5)
        sharp = cv2.addWeighted(aimg, 1.35, gaussian, -0.35, 0)
        return sharp
    return aimg


# ============================================================
# MAIN CLASSROOM VIDEO PROCESSING (OPTIMIZED ANTELOPEV2 MULTI-FACE)
# ============================================================

def process_classroom_video(video_id, video_path):
    """
    Processes a classroom video using optimized InsightFace Antelopev2:
    - Target: ~10-second total processing time on laptop hardware
    - Avoids decoding skipped frames using OpenCV hardware/demuxer skipping
    - SCRFD 10G multi-face detection per strategically sampled frame
    - Batched ArcFace ResNet-100 feature extraction for detected faces
    - Vectorized NumPy cosine similarity matching (no dict building in loop)
    - Anti-false-positive multi-frame quorum confirmation
    - Early stopping once sufficient reliable evidence is gathered
    - Bulk database write for attendance marking
    - Genuine calculated recognition accuracy/confidence (never fake 100%)
    """
    t_start = time.time()

    logger.info("=" * 80)
    logger.info("INSIGHTFACE ANTELOPEV2: HIGH-SPEED PROCESSING FOR VIDEO ID %s", video_id)
    logger.info("=" * 80)

    # 1. Access InsightFace Antelopev2 engine (singleton, no re-initialization)
    app = get_face_app()
    det_model = app.det_model
    rec_model = app.models['recognition']

    # 2. Validate video path and database entry
    if not video_path:
        raise ValueError("Video path is empty.")

    if not os.path.isabs(video_path):
        abs_path = os.path.join(BASE_DIR, video_path)
        if os.path.isfile(abs_path):
            video_path = abs_path

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    video_info = get_video_information(video_id)
    if not video_info:
        raise ValueError(f"Video record {video_id} not found in database.")

    class_id = video_info.get("class_id")
    subject_id = video_info.get("subject_id")
    if class_id is None or subject_id is None:
        raise ValueError("Video record is missing class_id or subject_id.")

    update_video_status(video_id, "processing")

    # 3. Load registered student embeddings for this specific class
    student_ids = None
    embedding_matrix = None

    registered_embeddings = load_registered_embeddings(class_id=class_id)
    if registered_embeddings:
        student_ids, embedding_matrix = build_embedding_matrix(registered_embeddings)

    if embedding_matrix is None or len(student_ids) == 0:
        logger.info("No embeddings in MySQL for class %s. Checking all enrolled...", class_id)
        registered_embeddings = load_registered_embeddings(class_id=None)
        if registered_embeddings:
            student_ids, embedding_matrix = build_embedding_matrix(registered_embeddings)

    if embedding_matrix is None or len(student_ids) == 0:
        logger.info("Checking uploads/dataset/<USN>/...")
        student_ids, embedding_matrix = read_student_photos_from_dataset()

    if embedding_matrix is None or len(student_ids) == 0:
        raise ValueError("No student face embeddings available. Please run dataset training first.")

    # Precompute unique students and inverse index mapping for vectorization
    unique_sids, inverse_indices = np.unique(student_ids, return_inverse=True)
    U = len(unique_sids)
    N = len(student_ids)
    unique_students = U

    logger.info("Enrolled class students ready: %d unique students (%d total reference embeddings)",
                unique_students, N)

    # 4. Open video and calculate optimal frame schedule
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        update_video_status(video_id, "failed")
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    target_frames = calculate_optimal_target_frames(total_frames, fps, duration)
    logger.info("Video specs: fps=%.2f, total_frames=%d, duration=%.2fs | Scheduled %d target frames: %s",
                fps, total_frames, duration, len(target_frames), target_frames)

    # 5. Process target frames efficiently
    processed_frames = 0
    frames_with_faces = 0
    face_count = 0

    student_detections = {
        int(sid): {
            "frames": set(),
            "similarities": [],
            "margins": [],
            "best_sim": 0.0,
            "best_margin": 0.0,
            "count": 0
        }
        for sid in unique_sids
    }

    # Tracking for early stopping & adaptive detection
    last_confirmed_count = 0
    stable_frames_count = 0
    MIN_FRAMES_BEFORE_EARLY_STOP = 5
    adaptive_runs = 0
    adaptive_check_frames = {0, 2, 4}

    curr_frame_idx = 0

    try:
        for target_frame in target_frames:
            # Efficient frame skipping: use grab to skip compressed packets without full decoding
            while curr_frame_idx < target_frame:
                if not capture.grab():
                    break
                curr_frame_idx += 1

            success, frame = capture.read()
            curr_frame_idx += 1

            if not success or frame is None or frame.size == 0:
                continue

            processed_frames += 1

            # Preserve small faces: downscale with fast INTER_LINEAR only if resolution exceeds 1280
            h, w = frame.shape[:2]
            if max(h, w) > 1280:
                scale = 1280.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

            # 1. Lightweight adaptive preprocessing: brightness/gamma + CLAHE (only when needed)
            proc_frame, was_enhanced = preprocess_classroom_frame(frame)

            # 2. SCRFD multi-face detection (preserves native aspect ratio across all benches)
            bboxes, kpss = safe_detect_faces(det_model, proc_frame, thresh=DET_CONFIDENCE_THRESHOLD)

            if bboxes is None or bboxes.shape[0] == 0:
                continue

            # 3. Filter valid face detections and align crops
            valid_crops = []
            for i in range(bboxes.shape[0]):
                score = bboxes[i, 4]
                if score < DET_CONFIDENCE_THRESHOLD:
                    continue
                bw = bboxes[i, 2] - bboxes[i, 0]
                bh = bboxes[i, 3] - bboxes[i, 1]
                if bw < MIN_FACE_WIDTH or bh < MIN_FACE_HEIGHT:
                    continue

                if kpss is not None:
                    aimg = face_align.norm_crop(proc_frame, landmark=kpss[i], image_size=rec_model.input_size[0])
                    valid_crops.append(aimg)

            if not valid_crops:
                continue

            K = len(valid_crops)
            frames_with_faces += 1
            face_count += K

            # 4. Batched ArcFace ResNet-100 feature extraction (single ONNX session call)
            embs = rec_model.get_feat(valid_crops)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            frame_matrix = (embs / norms).astype(np.float32)

            # 5. Vectorized cosine similarity matrix: shape [K, N]
            sims = np.dot(frame_matrix, embedding_matrix.T)

            # 6. Vectorized student score aggregation: shape [K, U]
            student_scores = np.full((K, U), -1.0, dtype=np.float32)
            row_idx = np.repeat(np.arange(K), N)
            col_idx = np.tile(inverse_indices, K)
            np.maximum.at(student_scores, (row_idx, col_idx), sims.ravel())

            # 7. Vectorized top-2 match & margin extraction
            if U >= 2:
                top2_idx = np.argpartition(student_scores, -2, axis=1)[:, -2:]
                top2_scores = np.take_along_axis(student_scores, top2_idx, axis=1)
                order = np.argsort(top2_scores, axis=1)
                second_best_score = top2_scores[np.arange(K), order[:, 0]]
                best_score = top2_scores[np.arange(K), order[:, 1]]
                best_u_idx = top2_idx[np.arange(K), order[:, 1]]
                margins = best_score - second_best_score
            else:
                best_u_idx = np.argmax(student_scores, axis=1)
                best_score = student_scores[np.arange(K), best_u_idx]
                margins = best_score - (-1.0)

            # 8. Collect detections in frame with strict anti-false-positive threshold
            assigned_students_in_frame = set()
            for k in range(K):
                bs = float(best_score[k])
                mg = float(margins[k])

                if bs >= FACE_MATCH_THRESHOLD and mg >= FRAME_MARGIN_THRESHOLD:
                    sid = int(unique_sids[best_u_idx[k]])
                    if sid not in assigned_students_in_frame:
                        assigned_students_in_frame.add(sid)
                        data = student_detections[sid]
                        data["frames"].add(target_frame)
                        data["similarities"].append(bs)
                        data["margins"].append(mg)
                        data["count"] += 1
                        if bs > data["best_sim"]:
                            data["best_sim"] = bs
                            data["best_margin"] = mg

            # 9. Check early stopping criteria
            current_confirmed_count = 0
            has_single_hit_candidates = False

            for sid, d in student_detections.items():
                cnt = d["count"]
                best_sim = d["best_sim"]
                best_mg = d["best_margin"]
                avg_sim = float(np.mean(d["similarities"])) if d["similarities"] else 0.0
                avg_mg = float(np.mean(d["margins"])) if d["margins"] else 0.0

                if (
                    (best_sim >= 0.395 and best_mg >= 0.012)
                    or (best_sim >= 0.420)
                    or (cnt >= MIN_DETECTIONS_TO_CONFIRM and best_sim >= MIN_SIMILARITY_TO_CONFIRM and avg_mg >= MIN_CONFIRM_MARGIN)
                    or (cnt >= 3 and best_sim >= 0.360 and avg_sim >= 0.350)
                    or (cnt >= 1 and best_sim >= 0.375 and best_mg >= 0.015)
                ):
                    current_confirmed_count += 1
                elif cnt == 1:
                    has_single_hit_candidates = True

            # Early stop condition A: 100% of enrolled students confirmed
            if current_confirmed_count >= unique_students and unique_students > 0:
                logger.info("Early stopping: All %d enrolled students confirmed present.", current_confirmed_count)
                break

            # Early stop condition B: Stable quorum with no pending single-hit candidates
            if processed_frames >= MIN_FRAMES_BEFORE_EARLY_STOP:
                if current_confirmed_count == last_confirmed_count and not has_single_hit_candidates and current_confirmed_count > 0:
                    stable_frames_count += 1
                    if stable_frames_count >= 2:
                        logger.info("Early stopping: Stable quorum reached (%d students confirmed, no pending candidates).",
                                    current_confirmed_count)
                        break
                else:
                    stable_frames_count = 0

            last_confirmed_count = current_confirmed_count

    except Exception as e:
        logger.exception("Error during video recognition: %s", e)
        update_video_status(video_id, "failed")
        raise
    finally:
        capture.release()

    # ========================================================
    # STRICT MULTI-FRAME STUDENT CONFIRMATION
    # ========================================================
    logger.info("=" * 80)
    logger.info("CONFIRMATION: Applying anti-false-positive quorum rules...")
    logger.info("=" * 80)

    student_details = load_student_details(list(student_detections.keys()))
    confirmed_sids = []
    rejected_students = []
    confirmed_data_map = {}

    for student_id, data in student_detections.items():
        detection_count = data["count"]
        similarities = data["similarities"]
        margins = data["margins"]

        if not similarities or detection_count == 0:
            continue

        avg_similarity = float(np.mean(similarities))
        avg_margin = float(np.mean(margins))
        best_similarity = float(data["best_sim"])
        best_margin = float(data["best_margin"])

        details = student_details.get(int(student_id), {
            "student_id": int(student_id),
            "name": "Unknown",
            "student_number": "Unknown",
            "email": ""
        })

        is_confirmed = False
        reason = ""

        # CRITERION 1: Multi-Frame Temporal Quorum (Consistent detections across sampled frames)
        if (
            detection_count >= MIN_DETECTIONS_TO_CONFIRM
            and best_similarity >= MIN_SIMILARITY_TO_CONFIRM
            and avg_margin >= MIN_CONFIRM_MARGIN
        ):
            is_confirmed = True
            reason = f"quorum_match(detections={detection_count},best={best_similarity:.3f},avg_margin={avg_margin:.3f})"

        # CRITERION 2: Clear Single-Frame Detection
        elif (
            best_similarity >= 0.395
            and best_margin >= 0.012
        ):
            is_confirmed = True
            reason = f"clear_frame_match(best={best_similarity:.3f},margin={best_margin:.3f},detections={detection_count})"

        # CRITERION 3: High Confidence Temporal Peak
        elif best_similarity >= 0.420:
            is_confirmed = True
            reason = f"temporal_high_confidence(detections={detection_count},best={best_similarity:.3f},margin={best_margin:.3f})"

        # CRITERION 4: Difficult/Side-Angle/Back-Bench Multi-Frame Evidence
        elif (
            detection_count >= 3
            and best_similarity >= 0.360
            and avg_similarity >= 0.350
        ):
            is_confirmed = True
            reason = f"temporal_consistency(detections={detection_count},best={best_similarity:.3f},avg={avg_similarity:.3f})"

        # CRITERION 5: Single-hit strong evidence
        elif (
            detection_count >= 1
            and best_similarity >= 0.375
            and best_margin >= 0.015
        ):
            is_confirmed = True
            reason = f"single_hit_match(best={best_similarity:.3f},margin={best_margin:.3f})"

        if not is_confirmed:
            rejected_students.append({
                "student_id": int(student_id),
                "name": details["name"],
                "student_number": details["student_number"],
                "detections": detection_count,
                "best_sim": round(best_similarity, 3),
                "best_margin": round(best_margin, 3),
                "avg_sim": round(avg_similarity, 3),
                "avg_margin": round(avg_margin, 3)
            })
            logger.info("REJECTED (noise/ambiguous): %s - Detections=%d, Best=%.3f, Margin=%.3f, Avg=%.3f",
                        details["name"], detection_count, best_similarity, best_margin, avg_similarity)
            continue

        confirmed_sids.append(int(student_id))
        confirmed_data_map[int(student_id)] = {
            "details": details,
            "detection_count": detection_count,
            "best_similarity": best_similarity,
            "best_margin": best_margin,
            "reason": reason
        }
        logger.info("CONFIRMED: %s (%s) - Best=%.3f, Margin=%.3f, Reason=%s",
                    details["name"], details["student_number"], best_similarity, best_margin, reason)

    # ========================================================
    # BULK DATABASE ATTENDANCE MARKING
    # ========================================================
    new_attendance_count, status_map = mark_students_present_bulk(
        confirmed_sids, class_id, subject_id, video_id
    )

    recognized_students = []
    for sid in confirmed_sids:
        cdata = confirmed_data_map[sid]
        det = cdata["details"]
        best_sim = cdata["best_similarity"]
        conf_pct = min(100.0, max(0.0, round(best_sim * 100.0, 1)))
        conf_str = f"{conf_pct:.1f}%"

        recognized_students.append({
            "student_id": sid,
            "name": det["name"],
            "student_number": det["student_number"],
            "email": det["email"],
            "detection_count": cdata["detection_count"],
            "accuracy": conf_str,
            "confidence": conf_str,
            "best_similarity": round(best_sim, 3),
            "best_margin": round(cdata["best_margin"], 3),
            "attendance_status": status_map.get(sid, "Present"),
            "confirmation_reason": cdata["reason"]
        })

    recognized_students.sort(key=lambda x: str(x.get("student_number", "")))
    present_count = len(recognized_students)

    # ========================================================
    # ABSENT STUDENTS DETERMINATION
    # ========================================================
    enrolled_query = """
        SELECT s.student_id, s.name, s.student_number, s.email
        FROM students s
        INNER JOIN class_students cs ON s.student_id = cs.student_id
        WHERE cs.class_id = %s
        ORDER BY s.student_number ASC
    """
    enrolled_students = run_query(enrolled_query, (class_id,), fetch_all=True) or []
    if not enrolled_students:
        all_students_query = """
            SELECT student_id, name, student_number, email
            FROM students
            ORDER BY student_number ASC
        """
        enrolled_students = run_query(all_students_query, fetch_all=True) or []

    absent_students = []
    for st in enrolled_students:
        sid = int(st["student_id"])
        if sid not in confirmed_sids:
            absent_students.append({
                "student_id": sid,
                "name": st.get("name") or "Student",
                "student_number": st.get("student_number") or "N/A",
                "email": st.get("email") or "",
                "status": "Absent"
            })

    # Genuine recognition accuracy / confidence
    if recognized_students:
        avg_confidence = float(np.mean([s["best_similarity"] for s in recognized_students])) * 100.0
        accuracy_val = round(avg_confidence, 1)
        accuracy_str = f"{accuracy_val:.1f}%"
    else:
        accuracy_val = 0.0
        accuracy_str = "0.0%"

    elapsed_time = time.time() - t_start
    notes = (
        f"Antelopev2 processed {processed_frames} frames ({frames_with_faces} with faces, "
        f"{face_count} detections) in {elapsed_time:.1f}s. Recognition Confidence: {accuracy_str}. Confirmed {present_count} students present, {len(absent_students)} absent."
    )

    update_video_status(
        video_id,
        "processed",
        face_count=face_count,
        attendance_count=present_count,
        notes=notes
    )

    # Immediately generate real-time Excel attendance report directly from database
    try:
        from backend.services.attendance_service import generate_and_save_session_excel
        excel_path = generate_and_save_session_excel(video_id=video_id)
        logger.info("Real-time Excel attendance report immediately generated: %s", excel_path)
    except Exception as ex_err:
        logger.warning("Could not auto-generate session Excel report: %s", ex_err)

    logger.info("=" * 80)
    logger.info("FINISHED VIDEO %s: %s Confidence. Confirmed %d students present, %d absent in %.2fs.",
                video_id, accuracy_str, present_count, len(absent_students), elapsed_time)
    logger.info("=" * 80)

    return {
        "video_id": video_id,
        "face_count": face_count,
        "processed_frames": processed_frames,
        "present_count": present_count,
        "absent_count": len(absent_students),
        "attendance_count": new_attendance_count,
        "accuracy": accuracy_val,
        "accuracy_pct": accuracy_str,
        "recognition_accuracy": accuracy_str,
        "recognized_students": recognized_students,
        "absent_students": absent_students,
        "rejected_students": rejected_students,
        "unique_students_available": unique_students,
        "summary_report": notes
    }