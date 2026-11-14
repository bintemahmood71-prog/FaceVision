
import cv2 # type: ignore
from recognition_engine import FaceRecognitionManager


manager = FaceRecognitionManager()

# STEP 1: REGISTER A PERSON

person_name = input("Enter person's name: ")

image_path = input(
    "Enter registration image path: "
)

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Registration image could not be loaded.")

try:
    saved_path = manager.register_face(image, person_name)
    print("Face registered successfully!")
    print("Saved at:", saved_path)

except Exception as error:
    print("Registration failed:", error)
    raise SystemExit

# STEP 2: TRAIN THE MODEL

try:
    total_samples, total_people = manager.train_model()

    print("\nTraining completed!")
    print("Registered people:", total_people)
    print("Training samples:", total_samples)

except Exception as error:
    print("Training failed:", error)
    raise SystemExit


# STEP 3: RECOGNIZE A FACE

test_path = input(
    "\nEnter image path for recognition: "
)

test_image = cv2.imread(test_path)

if test_image is None:
    raise FileNotFoundError("Test image could not be loaded.")

try:
    result_image, results = manager.recognize_faces(test_image)

    print("\nRecognition Results:")

    for index, result in enumerate(results, start=1):
        print(
            f"Face {index}: {result['name']} "
            f"(LBPH distance: {result['distance']})"
        )

    cv2.imwrite("recognition_result.jpg", result_image)

    print("\nResult saved as recognition_result.jpg")

except Exception as error:
    print("Recognition failed:", error)
