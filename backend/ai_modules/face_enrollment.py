# backend/ai_modules/face_enrollment.py

import os
import pickle
import logging
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FaceEnrollment:
    """
    Quality-aware AntelopeV2 face enrollment.

    The model is NOT changed.
    No student/USN is hardcoded.

    The enrollment stage keeps only useful face embeddings instead of
    treating every detected enrollment photo equally.
    """

    MODEL_NAME = "antelopev2"

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".jfif", ".webp", ".gif"
    }

    MIN_FACE_SIZE = 20

    # Keep enough different good samples for classroom matching.
    MAX_EMBEDDINGS_PER_STUDENT = 12

    # Very poor images are rejected.
    MIN_QUALITY = 0.28

    # Prefer a mixture of strong and reasonably different samples.
    MIN_EMBEDDING_DISTANCE = 0.035

    def __init__(self):
        print("=" * 70)
        print("INITIALIZING QUALITY-AWARE FACE ENROLLMENT")
        print("=" * 70)
        print(f"Model: {self.MODEL_NAME}")
        print("Provider: CPU")

        self.app = FaceAnalysis(
            name=self.MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("Face model loaded successfully.")
        print("=" * 70)

    def get_dataset_folder(self):
        folder = Config.DATASET_FOLDER

        # Check Config.DATASET_FOLDER first; if empty or missing, fallback to uploads/dataset
        if os.path.exists(folder):
            subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            if subdirs:
                return folder

        alt_folder = os.path.join(Config.BASE_DIR, "uploads", "dataset")
        if os.path.exists(alt_folder):
            subdirs = [d for d in os.listdir(alt_folder) if os.path.isdir(os.path.join(alt_folder, d))]
            if subdirs:
                return alt_folder

        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        return folder

    def get_embeddings_folder(self):
        folder = Config.EMBEDDINGS_FOLDER
        os.makedirs(folder, exist_ok=True)
        return folder

    def get_image_files(self, student_folder):
        image_files = []

        for filename in os.listdir(student_folder):
            filepath = os.path.join(
                student_folder,
                filename
            )

            if not os.path.isfile(filepath):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in self.SUPPORTED_EXTENSIONS:
                image_files.append(filepath)

        image_files.sort()
        return image_files

    def read_image(self, image_path):
        extension = os.path.splitext(
            image_path
        )[1].lower()

        if extension == ".gif":
            try:
                import imageio.v2 as imageio

                gif = imageio.get_reader(image_path)
                frame = gif.get_data(0)
                gif.close()

                if frame is None:
                    return None

                return cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

            except Exception as exc:
                logger.warning(
                    "Could not read GIF %s: %s",
                    image_path,
                    exc
                )
                return None

        return cv2.imread(image_path)

    @staticmethod
    def normalize_embedding(embedding):
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        norm = np.linalg.norm(embedding)

        if norm <= 0:
            return None

        return (
            embedding / norm
        ).astype(np.float32)

    @staticmethod
    def calculate_sharpness(image):
        if image is None or image.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            return float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            )

        except Exception:
            return 0.0

    def calculate_quality(
        self,
        image,
        face
    ):
        """
        General face-quality score.

        Main factors:
        - face size
        - sharpness
        - detector confidence

        This score is used only to choose better enrollment
        samples. It does not identify a student.
        """

        bbox = getattr(
            face,
            "bbox",
            None
        )

        if bbox is None:
            return 0.0

        x1, y1, x2, y2 = bbox

        width = max(
            0.0,
            float(x2 - x1)
        )

        height = max(
            0.0,
            float(y2 - y1)
        )

        if width < self.MIN_FACE_SIZE:
            return 0.0

        if height < self.MIN_FACE_SIZE:
            return 0.0

        face_area = width * height

        # Larger faces are more useful for recognition.
        size_score = min(
            face_area / 10000.0,
            1.0
        )

        # Sharpness is normalized conservatively.
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(image.shape[1], int(x2))
        iy2 = min(image.shape[0], int(y2))

        crop = image[iy1:iy2, ix1:ix2]

        sharpness = self.calculate_sharpness(
            crop
        )

        sharp_score = min(
            sharpness / 700.0,
            1.0
        )

        detector_score = float(
            getattr(
                face,
                "det_score",
                0.0
            )
        )

        detector_score = min(
            max(detector_score, 0.0),
            1.0
        )

        quality = (
            0.55 * size_score
            + 0.25 * sharp_score
            + 0.20 * detector_score
        )

        return float(
            max(0.0, min(quality, 1.0))
        )

    @staticmethod
    def embedding_distance(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        return float(
            1.0 - np.dot(a, b)
        )

    def select_quality_embeddings(
        self,
        candidates
    ):
        """
        Select strong but not identical enrollment samples.

        First choose the highest-quality image, then add other good
        images that are sufficiently different from already selected
        embeddings. This avoids filling the gallery with many nearly
        identical photos.
        """

        if not candidates:
            return []

        candidates = sorted(
            candidates,
            key=lambda x: x["quality"],
            reverse=True
        )

        selected = []

        for candidate in candidates:
            if len(selected) >= self.MAX_EMBEDDINGS_PER_STUDENT:
                break

            if candidate["quality"] < self.MIN_QUALITY:
                continue

            if not selected:
                selected.append(candidate)
                continue

            sufficiently_different = True

            for existing in selected:
                distance = self.embedding_distance(
                    candidate["embedding"],
                    existing["embedding"]
                )

                if distance < self.MIN_EMBEDDING_DISTANCE:
                    sufficiently_different = False
                    break

            if sufficiently_different:
                selected.append(candidate)

        # If filtering was too aggressive, keep a few additional good
        # samples so students are not left with only one embedding.
        if len(selected) < 3:
            for candidate in candidates:
                if len(selected) >= min(
                    3,
                    self.MAX_EMBEDDINGS_PER_STUDENT
                ):
                    break

                if candidate["quality"] < self.MIN_QUALITY:
                    continue

                already_selected = any(
                    candidate["filename"] == item["filename"]
                    for item in selected
                )

                if not already_selected:
                    selected.append(candidate)

        return selected

    def process_photo(
        self,
        image_path,
        expected_usn
    ):
        image = self.read_image(image_path)

        if image is None:
            return None, "Could not read image"

        try:
            faces = self.app.get(image)

        except Exception as exc:
            return None, f"Face detection failed: {exc}"

        if not faces:
            return None, "No face detected"

        if len(faces) != 1:
            return None, (
                f"Expected 1 face, found {len(faces)}"
            )

        face = faces[0]

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        face_width = x2 - x1
        face_height = y2 - y1

        if (
            face_width < self.MIN_FACE_SIZE
            or face_height < self.MIN_FACE_SIZE
        ):
            return None, (
                f"Face too small "
                f"({face_width}x{face_height})"
            )

        embedding = getattr(
            face,
            "embedding",
            None
        )

        if embedding is None:
            return None, "Embedding not generated"

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return None, "Invalid embedding"

        quality = self.calculate_quality(
            image,
            face
        )

        return {
            "embedding": embedding,
            "quality": quality,
            "face_width": face_width,
            "face_height": face_height,
            "det_score": float(
                getattr(face, "det_score", 0.0)
            )
        }, None

    def enroll_student(
        self,
        student_folder,
        embeddings_folder
    ):
        usn = os.path.basename(
            os.path.normpath(student_folder)
        )

        print()
        print("-" * 70)
        print(f"Student: {usn}")
        print("-" * 70)

        image_files = self.get_image_files(
            student_folder
        )

        if not image_files:
            print("No supported photos found.")

            return {
                "usn": usn,
                "success": False,
                "photos_processed": 0,
                "embeddings_created": 0,
                "photos_failed": 0
            }

        candidates = []
        failed_count = 0

        for index, image_path in enumerate(
            image_files,
            start=1
        ):
            filename = os.path.basename(
                image_path
            )

            print(
                f"  [{index}/{len(image_files)}] "
                f"{filename}"
            )

            result, error = self.process_photo(
                image_path,
                usn
            )

            if result is None:
                failed_count += 1
                print(
                    f"       FAILED: {error}"
                )
                continue

            candidates.append({
                "embedding": result["embedding"],
                "quality": result["quality"],
                "filename": filename,
                "face_width": result["face_width"],
                "face_height": result["face_height"],
                "det_score": result["det_score"]
            })

            print(
                f"       OK "
                f"(quality={result['quality']:.3f}, "
                f"face={result['face_width']}x"
                f"{result['face_height']})"
            )

        selected = self.select_quality_embeddings(
            candidates
        )

        if not selected:
            print("No sufficiently good face embeddings.")

            return {
                "usn": usn,
                "success": False,
                "photos_processed": len(image_files),
                "embeddings_created": 0,
                "photos_failed": failed_count
            }

        output_path = os.path.join(
            embeddings_folder,
            f"{usn}.pkl"
        )

        embeddings = [
            item["embedding"]
            for item in selected
        ]

        photo_names = [
            item["filename"]
            for item in selected
        ]

        qualities = [
            float(item["quality"])
            for item in selected
        ]

        data = {
            "usn": usn,
            "embeddings": embeddings,
            "photo_names": photo_names,
            "qualities": qualities,
            "embedding_quality": qualities,
            "embedding_count": len(embeddings),
            "model": self.MODEL_NAME,
            "quality_aware": True
        }

        with open(
            output_path,
            "wb"
        ) as file:
            pickle.dump(
                data,
                file
            )

        # Synchronize centroid to MySQL database if available
        try:
            from backend.services.db_service import run_query
            centroid = np.mean(embeddings, axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
            centroid_bytes = centroid_norm.astype(np.float32).tobytes()
            run_query(
                "UPDATE students SET face_encoding = %s WHERE student_number = %s",
                (centroid_bytes, usn),
                commit=True
            )
        except Exception:
            pass

        print()
        print(
            f"  Selected: {len(selected)} / "
            f"{len(candidates)} usable embeddings"
        )

        print(
            f"  Quality range: "
            f"{min(qualities):.3f} - "
            f"{max(qualities):.3f}"
        )

        print(
            f"  Output: {output_path}"
        )

        return {
            "usn": usn,
            "success": True,
            "photos_processed": len(image_files),
            "embeddings_created": len(embeddings),
            "photos_failed": failed_count
        }

    def enroll_all(self):
        dataset_folder = self.get_dataset_folder()
        embeddings_folder = self.get_embeddings_folder()

        print()
        print("=" * 70)
        print("STARTING QUALITY-AWARE FACE ENROLLMENT")
        print("=" * 70)
        print(f"Dataset: {dataset_folder}")
        print(f"Embeddings: {embeddings_folder}")
        print(f"Model: {self.MODEL_NAME}")
        print()

        student_folders = []

        for name in sorted(
            os.listdir(dataset_folder)
        ):
            path = os.path.join(
                dataset_folder,
                name
            )

            if os.path.isdir(path):
                student_folders.append(path)

        print(
            f"Students found: "
            f"{len(student_folders)}"
        )

        students_enrolled = 0
        students_failed = 0
        total_photos = 0
        total_embeddings = 0
        total_failed_photos = 0

        for student_folder in student_folders:

            result = self.enroll_student(
                student_folder,
                embeddings_folder
            )

            total_photos += result[
                "photos_processed"
            ]

            total_embeddings += result[
                "embeddings_created"
            ]

            total_failed_photos += result[
                "photos_failed"
            ]

            if result["success"]:
                students_enrolled += 1
            else:
                students_failed += 1

        print()
        print("=" * 70)
        print("QUALITY-AWARE FACE ENROLLMENT COMPLETED")
        print("=" * 70)

        print(
            f"Students found      : "
            f"{len(student_folders)}"
        )

        print(
            f"Students enrolled   : "
            f"{students_enrolled}"
        )

        print(
            f"Students failed     : "
            f"{students_failed}"
        )

        print(
            f"Photos processed    : "
            f"{total_photos}"
        )

        print(
            f"Embeddings created  : "
            f"{total_embeddings}"
        )

        print(
            f"Photos failed       : "
            f"{total_failed_photos}"
        )

        print(
            f"Embedding folder    : "
            f"{embeddings_folder}"
        )

        print(
            f"Model used          : "
            f"{self.MODEL_NAME}"
        )

        print("=" * 70)

        return {
            "students_found": len(student_folders),
            "students_enrolled": students_enrolled,
            "students_failed": students_failed,
            "photos_processed": total_photos,
            "embeddings_created": total_embeddings,
            "photos_failed": total_failed_photos
        }


if __name__ == "__main__":
    try:
        enrollment = FaceEnrollment()
        enrollment.enroll_all()

    except KeyboardInterrupt:
        print()
        print("Enrollment interrupted by user.")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ENROLLMENT ERROR")
        print("=" * 70)
        print(exc)
        print("=" * 70)
