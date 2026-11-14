
import cv2
import os


class FaceDetector:
    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_alt2.xml"
        )

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Face detection model not found: {cascade_path}"
            )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(
                "Could not load the Haar Cascade model."
            )

    def detect_faces(self, image):
        if image is None:
            raise ValueError("The input image is empty or invalid.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        annotated_image = image.copy()

        for (x, y, width, height) in faces:
            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + width, y + height),
                (0, 200, 100),
                2
            )

        return annotated_image, faces

    def get_face_count(self, faces):
        return len(faces)
