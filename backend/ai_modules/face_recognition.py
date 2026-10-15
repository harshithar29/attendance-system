# backend/ai_modules/face_recognition.py

import os
import pickle
import logging
import numpy as np
import cv2

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align

from config.config import Config


logger = logging.getLogger(__name__)


class FaceRecognitionService:
    MODEL_NAME = "antelopev2"

    DETECTION_SIZE = (640, 640)

    MIN_FACE_SIZE = 15

    MATCH_THRESHOLD = 0.40
    MIN_MATCH_MARGIN = 0.025

    _initialized = False
    _detector = None
    _recognizer = None

    _gallery_matrix = None
    _gallery_usns = None
    _student_embeddings = {}

    @classmethod
    def _initialize_model(cls):
        if cls._initialized:
            return

        logger.info("Loading antelopev2 detection model...")

        cls._detector = FaceAnalysis(
            name=cls.MODEL_NAME,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"]
        )

        cls._detector.prepare(
            ctx_id=0,
            det_size=cls.DETECTION_SIZE
        )

        logger.info("Loading antelopev2 recognition model...")

        model_dir = os.path.expanduser(
            os.path.join(
                "~",
                ".insightface",
                "models",
                "antelopev2"
            )
        )

        recognizer_path = os.path.join(
            model_dir,
            "glintr100.onnx"
        )

        if not os.path.exists(recognizer_path):
            alt_candidates = [
                os.path.join(Config.BASE_DIR, "backend", "models", "glintr100.onnx"),
                os.path.join(Config.BASE_DIR, "models", "glintr100.onnx"),
            ]
            for ac in alt_candidates:
                if os.path.exists(ac):
                    recognizer_path = ac
                    break

        try:
            cls._recognizer = get_model(
                recognizer_path,
                providers=["CPUExecutionProvider"]
            )
        except Exception:
            try:
                cls._recognizer = get_model(
                    "glintr100",
                    providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                logger.warning("Could not load recognizer model directly: %s", e)
                cls._recognizer = None

        if cls._recognizer is not None:
            cls._recognizer.prepare(ctx_id=0)

        cls._load_gallery()

        cls._initialized = True

        print("AntelopeV2 detection + recognition models loaded.")

    # ---------------------------------------------------------
    # GALLERY
    # ---------------------------------------------------------

    @classmethod
    def _load_gallery(cls):
        print("Loading face embedding gallery...")

        embeddings_folder = Config.EMBEDDINGS_FOLDER

        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder, exist_ok=True)

        all_embeddings = []
        all_usns = []

        cls._student_embeddings = {}

        for filename in sorted(os.listdir(embeddings_folder)):
            if not filename.lower().endswith(".pkl"):
                continue

            path = os.path.join(
                embeddings_folder,
                filename
            )

            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)

                usn = str(
                    data.get("usn")
                    or os.path.splitext(filename)[0]
                )

                embeddings = data.get("embeddings")

                # Backward compatibility
                if embeddings is None:
                    single = data.get("embedding")

                    if single is not None:
                        embeddings = [single]

                if not embeddings:
                    continue

                student_list = []

                for emb in embeddings:
                    emb = np.asarray(
                        emb,
                        dtype=np.float32
                    ).flatten()

                    norm = np.linalg.norm(emb)

                    if norm == 0:
                        continue

                    emb = emb / norm

                    all_embeddings.append(emb)
                    all_usns.append(usn)
                    student_list.append(emb)

                if student_list:
                    cls._student_embeddings[usn] = student_list

            except Exception as e:
                logger.warning(
                    "Could not load %s: %s",
                    filename,
                    e
                )

        # Fallback from database if no pkl files exist yet
        if not all_embeddings:
            try:
                from backend.services.db_service import run_query
                db_rows = run_query(
                    "SELECT s.student_number, fe.embedding FROM face_embeddings fe JOIN students s ON fe.student_id = s.student_id WHERE fe.embedding IS NOT NULL",
                    fetch_all=True
                ) or []
                for r in db_rows:
                    usn = str(r.get("student_number") or "").strip().upper()
                    raw = r.get("embedding")
                    if raw:
                        emb = np.frombuffer(raw, dtype=np.float32).flatten()
                        if emb.size == 512:
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                                all_embeddings.append(emb)
                                all_usns.append(usn)
                                cls._student_embeddings.setdefault(usn, []).append(emb)
            except Exception as dberr:
                logger.debug("Database gallery fallback: %s", dberr)

        if not all_embeddings:
            logger.warning("No face embeddings found in gallery.")
            return

        cls._gallery_matrix = np.asarray(
            all_embeddings,
            dtype=np.float32
        )

        cls._gallery_usns = np.asarray(
            all_usns
        )

        print(
            f"Students in gallery : "
            f"{len(cls._student_embeddings)}"
        )

        print(
            f"Embeddings in gallery: "
            f"{len(cls._gallery_matrix)}"
        )

        print(
            f"Gallery model        : "
            f"{cls.MODEL_NAME}"
        )

        print("Face embedding gallery loaded.")

    # ---------------------------------------------------------
    # DETECTION ONLY
    # ---------------------------------------------------------

    @classmethod
    def detect_faces(cls, frame):
        """
        Fast detection-only operation.

        IMPORTANT:
        This does NOT calculate recognition embeddings.
        """

        cls._initialize_model()

        if frame is None:
            return []

        try:
            faces = cls._detector.get(frame)

            valid_faces = []

            for face in faces:
                bbox = face.bbox.astype(int)

                x1, y1, x2, y2 = bbox

                width = x2 - x1
                height = y2 - y1

                if width < cls.MIN_FACE_SIZE:
                    continue

                if height < cls.MIN_FACE_SIZE:
                    continue

                if face.kps is None:
                    continue

                valid_faces.append(face)

            return valid_faces

        except Exception as e:
            logger.error(
                "Face detection error: %s",
                e
            )
            return []

    # ---------------------------------------------------------
    # SINGLE FACE RECOGNITION
    # ---------------------------------------------------------

    @classmethod
    def recognize_face_object(cls, frame, face):
        """
        Recognize an already detected face.

        Detection is NOT repeated.
        """

        cls._initialize_model()

        try:
            if face.kps is None:
                return None

            # Align the detected face using landmarks.
            aligned = face_align.norm_crop(
                frame,
                landmark=face.kps,
                image_size=112
            )

            # Run recognition model only once.
            if cls._recognizer is None:
                return None

            embedding = cls._recognizer.get_feat(
                aligned
            )

            if embedding is None:
                return None

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).flatten()

            norm = np.linalg.norm(embedding)

            if norm == 0:
                return None

            embedding /= norm

            return cls._best_student_match(
                embedding
            )

        except Exception as e:
            logger.debug(
                "Face recognition error: %s",
                e
            )
            return None

    # ---------------------------------------------------------
    # MATCHING
    # ---------------------------------------------------------

    @classmethod
    def _best_student_match(cls, embedding):
        if cls._gallery_matrix is None or len(cls._gallery_matrix) == 0:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        similarities = np.dot(
            cls._gallery_matrix,
            embedding
        )

        student_scores = {}

        for index, score in enumerate(similarities):
            usn = cls._gallery_usns[index]

            if (
                usn not in student_scores
                or score > student_scores[usn]
            ):
                student_scores[usn] = float(score)

        if not student_scores:
            return None

        ranked = sorted(
            student_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_usn, best_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        margin = best_score - second_score

        status = "unknown"

        if (
            best_score >= cls.MATCH_THRESHOLD
            and margin >= cls.MIN_MATCH_MARGIN
        ):
            status = "recognized"

        return {
            "usn": best_usn,
            "score": round(best_score, 4),
            "similarity": round(best_score, 4),
            "margin": round(margin, 4),
            "status": status
        }

    # ---------------------------------------------------------
    # COMPATIBILITY METHOD
    # ---------------------------------------------------------

    @classmethod
    def recognize_faces(cls, frame):
        faces = cls.detect_faces(frame)

        results = []

        for face in faces:
            result = cls.recognize_face_object(
                frame,
                face
            )

            if result is None:
                continue

            bbox = face.bbox

            result["face_width"] = float(
                bbox[2] - bbox[0]
            )

            result["face_height"] = float(
                bbox[3] - bbox[1]
            )

            results.append(result)

        return results

    @classmethod
    def recognize_single_face(cls, frame):
        results = cls.recognize_faces(frame)

        recognized = [
            r for r in results
            if r.get("status") == "recognized"
        ]

        if not recognized:
            return None

        return max(
            recognized,
            key=lambda r: r["score"]
        )


# -------------------------------------------------------------
# DIRECT TEST
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("FACE RECOGNITION SERVICE TEST")
    print("=" * 70)

    FaceRecognitionService._initialize_model()

    print()
    print(
        "Students :",
        len(FaceRecognitionService._student_embeddings)
    )

    print(
        "Embeddings:",
        len(FaceRecognitionService._gallery_matrix) if FaceRecognitionService._gallery_matrix is not None else 0
    )

    print()
    print("Service ready.")
