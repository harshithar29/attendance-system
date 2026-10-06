"""
scripts/register_students.py

Register student face images into the attendance system.

Filename examples:

    4GW23CI019.jpg
    4GW23CI020.jpeg
    4GW23CI028.gif
    4GW23CI059.jfif

The filename without extension is used as the USN.
"""

import os
import sys

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from backend.services.face_service import (
    generate_embedding
)

from backend.services.db_service import (
    run_query
)


# ============================================================
# CONFIGURATION
# ============================================================

PHOTO_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "uploads",
    "student_photos"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".jfif",
    ".webp"
}


# ============================================================
# IMAGE CHECK
# ============================================================

def is_supported_image(
    filename
):
    """
    Check whether a file is a supported image.
    """

    extension = os.path.splitext(
        filename
    )[1].lower()

    return (
        extension
        in SUPPORTED_EXTENSIONS
    )


# ============================================================
# EXTRACT STUDENT NUMBER
# ============================================================

def extract_student_number(
    filename
):
    """
    Extract USN from filename.

    Example:

        4GW23CI019.jpg

    returns:

        4GW23CI019
    """

    base_name = os.path.basename(
        filename
    )

    student_number = os.path.splitext(
        base_name
    )[0].strip()

    if not student_number:
        raise ValueError(
            f"Unable to extract student number from {filename}"
        )

    return student_number


# ============================================================
# FIND STUDENT
# ============================================================

def find_student(
    student_number
):
    """
    Find student using USN.
    """

    query = """
        SELECT
            student_id,
            usn
        FROM students
        WHERE usn = %s
        LIMIT 1
    """

    return run_query(
        query,
        (student_number,),
        fetch_one=True
    )


# ============================================================
# CREATE STUDENT
# ============================================================

def create_student(
    student_number
):
    """
    Create a student record.
    """

    query = """
        INSERT INTO students
        (
            usn
        )
        VALUES
        (
            %s
        )
    """

    run_query(
        query,
        (student_number,),
        commit=True
    )

    result = find_student(
        student_number
    )

    if not result:

        raise ValueError(
            "Student was created but could not be retrieved."
        )

    return result[
        "student_id"
    ]


# ============================================================
# SAVE EMBEDDING
# ============================================================

def save_embedding(
    student_id,
    image_path,
    embedding
):
    """
    Save student embedding.

    Existing embeddings for this student are removed first
    so that the newest registration becomes the active one.
    """

    delete_query = """
        DELETE FROM face_embeddings
        WHERE student_id = %s
    """

    run_query(
        delete_query,
        (student_id,),
        commit=True
    )

    insert_query = """
        INSERT INTO face_embeddings
        (
            student_id,
            image_path,
            embedding
        )
        VALUES
        (
            %s,
            %s,
            %s
        )
    """

    import pickle

    serialized_embedding = pickle.dumps(
        embedding
    )

    run_query(
        insert_query,
        (
            student_id,
            image_path,
            serialized_embedding
        ),
        commit=True
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_student_image(
    filename
):

    student_number = (
        extract_student_number(
            filename
        )
    )

    image_path = os.path.join(
        PHOTO_DIRECTORY,
        filename
    )

    print()
    print("=" * 60)

    print(
        f"Processing: {filename}"
    )

    print(
        f"Student No: {student_number}"
    )

    # --------------------------------------------------------
    # FIND OR CREATE STUDENT
    # --------------------------------------------------------

    student = find_student(
        student_number
    )

    if student:

        student_id = student[
            "student_id"
        ]

        print(
            "Existing student found: "
            f"ID={student_id}"
        )

    else:

        student_id = create_student(
            student_number
        )

        print(
            "Created student: "
            f"ID={student_id}"
        )

    # --------------------------------------------------------
    # GENERATE EMBEDDING
    # --------------------------------------------------------

    print(
        "Detecting face and generating "
        "FaceNet embedding..."
    )

    embedding = generate_embedding(
        image_path
    )

    if embedding is None:

        raise ValueError(
            "Embedding generation returned None."
        )

    print(
        "Embedding generated successfully "
        f"({len(embedding)} dimensions)"
    )

    # --------------------------------------------------------
    # RELATIVE IMAGE PATH
    # --------------------------------------------------------

    relative_path = os.path.relpath(
        image_path,
        PROJECT_ROOT
    ).replace(
        "\\",
        "/"
    )

    # --------------------------------------------------------
    # SAVE EMBEDDING
    # --------------------------------------------------------

    save_embedding(
        student_id,
        relative_path,
        embedding
    )

    print(
        "✓ Fresh embedding saved successfully."
    )

    return "success"


# ============================================================
# GET UNIQUE STUDENT IMAGES
# ============================================================

def get_unique_student_images():
    """
    Find supported images.

    Only one image is selected for each USN.
    """

    files = sorted(
        os.listdir(
            PHOTO_DIRECTORY
        )
    )

    student_images = {}

    duplicate_files = []

    for filename in files:

        full_path = os.path.join(
            PHOTO_DIRECTORY,
            filename
        )

        # ----------------------------------------------------
        # FILE CHECK
        # ----------------------------------------------------

        if not os.path.isfile(
            full_path
        ):
            continue

        # ----------------------------------------------------
        # EXTENSION CHECK
        # ----------------------------------------------------

        if not is_supported_image(
            filename
        ):
            continue

        student_number = (
            extract_student_number(
                filename
            )
        )

        # ----------------------------------------------------
        # FIRST IMAGE WINS
        # ----------------------------------------------------

        if (
            student_number
            not in student_images
        ):

            student_images[
                student_number
            ] = filename

        else:

            duplicate_files.append(
                (
                    student_number,
                    filename,
                    student_images[
                        student_number
                    ]
                )
            )

    return (
        sorted(
            student_images.values()
        ),
        duplicate_files
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GSSS ATTENDANCE PORTAL")
    print("STUDENT FACE REGISTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK DIRECTORY
    # --------------------------------------------------------

    if not os.path.exists(
        PHOTO_DIRECTORY
    ):

        print()

        print(
            "ERROR: Student photo directory "
            "does not exist:"
        )

        print(
            PHOTO_DIRECTORY
        )

        return

    # --------------------------------------------------------
    # FIND UNIQUE IMAGES
    # --------------------------------------------------------

    (
        image_files,
        duplicate_files
    ) = get_unique_student_images()

    print()

    print(
        "Unique students/images found: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # SHOW DUPLICATES
    # --------------------------------------------------------

    if duplicate_files:

        print()

        print(
            "DUPLICATE USN IMAGE FILES FOUND"
        )

        print("-" * 70)

        for (
            student_number,
            skipped_file,
            selected_file
        ) in duplicate_files:

            print(
                f"USN: {student_number}"
            )

            print(
                f"  Using   : {selected_file}"
            )

            print(
                f"  Skipping: {skipped_file}"
            )

    if not image_files:

        print()

        print(
            "No supported student images found."
        )

        return

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    success_count = 0

    failed_count = 0

    failed_files = []

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for filename in image_files:

        try:

            result = (
                process_student_image(
                    filename
                )
            )

            if result == "success":

                success_count += 1

        except Exception as error:

            failed_count += 1

            failed_files.append(
                (
                    filename,
                    str(error)
                )
            )

            print()

            print(
                f"✗ FAILED: {filename}"
            )

            print(
                f"Reason: {error}"
            )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print()

    print("=" * 70)
    print("REGISTRATION COMPLETE")
    print("=" * 70)

    print(
        f"Unique student images : "
        f"{len(image_files)}"
    )

    print(
        f"Successfully registered: "
        f"{success_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print(
        f"Duplicate image files skipped: "
        f"{len(duplicate_files)}"
    )

    # --------------------------------------------------------
    # FAILED FILES
    # --------------------------------------------------------

    if failed_files:

        print()

        print(
            "FILES REQUIRING REVIEW"
        )

        print("-" * 70)

        for (
            filename,
            reason
        ) in failed_files:

            print()

            print(
                filename
            )

            print(
                f"  {reason}"
            )

    print()

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()