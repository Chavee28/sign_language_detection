import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

camera_index = 0
cap = cv2.VideoCapture(camera_index)


if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}")
    exit()

for j in range(number_of_classes):
    # Create a directory for each class
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.putText(frame, 'Ready? Press "Q" to start collecting!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        print(f"Captured image {counter}/{dataset_size} for class {j}")

        # Check if user wants to stop the collection early by pressing 's'
        if key & 0xFF == ord('s'):
            print(f"Stopping early. Collected {counter} images for class {j}.")
            break

cap.release()
cv2.destroyAllWindows()
