
import cv2 # type: ignore
from face_engine import FaceDetector

# Change this to the name of your test image.
IMAGE_PATH = "test_face.jpg"

# Load the image.
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}. "
        "Check that the image is in the project folder."
    )

# Initialize the detector.
detector = FaceDetector()

# Detect faces.
result_image, detected_faces = detector.detect_faces(image)

# Display the results.
print("Detection completed.")
print("Number of faces detected:", detector.get_face_count(detected_faces))

for index, (x, y, width, height) in enumerate(detected_faces, start=1):
    print(
        f"Face {index}: "
        f"x={x}, y={y}, width={width}, height={height}"
    )

# Save the output image.
cv2.imwrite("detection_result.jpg", result_image)

print("Result saved as detection_result.jpg")
