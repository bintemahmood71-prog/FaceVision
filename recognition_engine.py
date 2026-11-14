
import cv2 # type: ignore
import json
import re
import uuid

from pathlib import Path
from face_engine import FaceDetector


class FaceRecognitionManager:

    def __init__(self):
        self.detector = FaceDetector()

        self.base_dir = Path("data")
        self.faces_dir = self.base_dir / "faces"
        self.models_dir = self.base_dir / "models"

        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.models_dir / "trainer.yml"
        self.labels_path = self.models_dir / "labels.json"

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.labels = {}

    def register_face(self, image, person_name):
        """Detect and save one face sample for a person."""

        person_name = person_name.strip()

        if not person_name or person_name in (".", ".."):
            raise ValueError("Please enter a valid name.")

        if "/" in person_name or "\\" in person_name:
            raise ValueError("Name cannot contain path separators.")

        if not re.fullmatch(r"[\w -]+", person_name):
            raise ValueError(
                "Use letters, numbers, spaces, hyphens or underscores."
            )

        if image is None:
            raise ValueError("Invalid image.")

        annotated_image, faces = self.detector.detect_faces(image)

        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly one face, but detected {len(faces)}."
            )

        x, y, width, height = faces[0]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face_crop = gray[y:y + height, x:x + width]
        face_crop = cv2.resize(face_crop, (200, 200))

        person_folder = self.faces_dir / person_name
        person_folder.mkdir(parents=True, exist_ok=True)

        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = person_folder / image_name

        saved = cv2.imwrite(str(image_path), face_crop)

        if not saved:
            raise IOError("Could not save the face sample.")

        return str(image_path)

    def train_model(self):
        """Train LBPH using all registered face samples."""

        training_faces = []
        training_labels = []
        self.labels = {}

        person_folders = sorted(
            folder for folder in self.faces_dir.iterdir()
            if folder.is_dir()
        )

        label_id = 0

        for person_folder in person_folders:

            samples = list(person_folder.glob("*.jpg"))

            if not samples:
                continue

            self.labels[label_id] = person_folder.name

            for sample_path in samples:

                image = cv2.imread(
                    str(sample_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    continue

                image = cv2.resize(image, (200, 200))

                training_faces.append(image)
                training_labels.append(label_id)

            label_id += 1

        if not training_faces:
            raise ValueError(
                "No registered face samples found. Register faces first."
            )

        self.recognizer.train(
            training_faces,
            __import__("numpy").array(training_labels, dtype="int32")
        )

        self.recognizer.save(str(self.model_path))

        with open(self.labels_path, "w", encoding="utf-8") as file:
            json.dump(self.labels, file, indent=4)

        return len(training_faces), len(self.labels)

    def load_model(self):
        """Load the previously trained model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model not found. Train the model first."
            )

        if not self.labels_path.exists():
            raise FileNotFoundError("Label file not found.")

        self.recognizer.read(str(self.model_path))

        with open(self.labels_path, "r", encoding="utf-8") as file:
            saved_labels = json.load(file)

        self.labels = {
            int(key): value for key, value in saved_labels.items()
        }

    def recognize_faces(self, image, threshold=70):
        """
        Recognize faces in an image.

        Lower LBPH distance means a closer match.
        Threshold is an experimental cutoff, not a probability.
        """

        if image is None:
            raise ValueError("Invalid input image.")

        self.load_model()

        annotated_image, faces = self.detector.detect_faces(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = []

        for (x, y, width, height) in faces:

            face_crop = gray[y:y + height, x:x + width]
            face_crop = cv2.resize(face_crop, (200, 200))

            predicted_label, distance = self.recognizer.predict(face_crop)

            if (
                predicted_label in self.labels
                and distance <= threshold
            ):
                name = self.labels[predicted_label]
            else:
                name = "Unknown"

            results.append({
                "name": name,
                "distance": round(float(distance), 2),
                "box": (int(x), int(y), int(width), int(height))
            })

            color = (0, 200, 100) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + width, y + height),
                color,
                2
            )

            cv2.putText(
                annotated_image,
                name,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return annotated_image, results
