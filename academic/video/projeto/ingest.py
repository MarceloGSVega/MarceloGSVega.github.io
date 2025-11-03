import cv2
import time
import os

url = "https://cameras.santoandre.sp.gov.br/coi04/ID_573"
camera_id = 655
outra_camera = 573
OUTPUT_DIR = "imagens"
INTERVAL_SECONDS = 20  # Save one screenshot every 20 seconds

cap = cv2.VideoCapture(url)


last_saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    # Display the frame in a local window
    cv2.imshow("Avenida dos Estados", frame)

    # Save a screenshot every INTERVAL_SECONDS
    current_time = time.time()
    if current_time - last_saved >= INTERVAL_SECONDS:
        filename = os.path.join(OUTPUT_DIR, f"screenshot_{camera_id}_{int(current_time)}.jpg")
        cv2.imwrite(filename, frame)
        last_saved = current_time

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
