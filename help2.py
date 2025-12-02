import os
import re
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# ------------------ SIMPLE HELPER: CLEAN TEXT ------------------ #

def clean_plate_text(text: str) -> str:
    """
    Make OCR text cleaner:
    - convert to uppercase
    - keep only A–Z and 0–9
    - ignore too short / too long text
    """
    text = text.upper()

    # remove everything that is NOT A–Z or 0–9
    text = re.sub(r'[^A-Z0-9]', '', text)

    # simple length check (you can change this later)
    if len(text) < 4 or len(text) > 12:
        return ""

    return text


# ------------------ MAIN FUNCTION ------------------ #

def process_video(video_path, model_path, output_video_path="output.mp4", csv_path="plates.csv"):
    """
    1) Open video
    2) Use YOLO to detect number plates
    3) Crop plate area and use EasyOCR to read text
    4) Draw box + text on video
    5) Save unique plate numbers to CSV
    6) Show a separate window with all detected plate numbers
    """

    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    # 2. Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)

    # 3. Initialize EasyOCR
    print("Starting EasyOCR...")
    reader = easyocr.Reader(['en'])

    # 4. Prepare video writer (for saving output video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Output video will be saved as:", output_video_path)

    # 5. Prepare CSV file for saving plate numbers
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["plate_text"])  # header row
    print("Detected plates will be saved in:", csv_path)

    # To avoid duplicate plates
    seen_plates = set()

    # ------------------ READ FRAMES IN A LOOP ------------------ #

    while True:
        # Read one frame from video
        ret, frame = cap.read()

        if not ret:
            print("End of video or cannot read frame.")
            break

        # Run YOLO on this frame
        # conf=0.4 means "ignore detections with confidence < 0.4"
        results = model(frame, conf=0.4)

        # Go through each detection result
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Get all bounding boxes as a NumPy array
            boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Make sure box is inside image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                # Crop the plate region from the frame
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size == 0:
                    continue

                # OPTIONAL small preprocessing: convert to gray
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                # EasyOCR can work with grayscale or color images
                ocr_result = reader.readtext(
                    gray_plate, detail=1, paragraph=False)

                plate_text = ""

                # Loop through OCR results
                for (bbox, raw_text, conf) in ocr_result:
                    cleaned_text = clean_plate_text(raw_text)
                    if cleaned_text:
                        plate_text = cleaned_text
                        break  # take the first valid one

                # Draw rectangle around detected area
                if plate_text:
                    color = (0, 255, 0)  # green if text detected
                else:
                    color = (0, 0, 255)  # red if no text

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Put plate text (or "Plate") above the box
                label = plate_text if plate_text else "Plate"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # If we got a valid plate text and it's new, save it
                if plate_text and plate_text not in seen_plates:
                    seen_plates.add(plate_text)
                    csv_writer.writerow([plate_text])
                    csv_file.flush()
                    print("New plate detected:", plate_text)

        # ------------------ SEPARATE WINDOW: DETECTED PLATES LIST ------------------ #

        # Create a black image to display plate list
        plate_list_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(
            plate_list_img,
            "Detected Plates:",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y = 80
        # show last 10 plates (or fewer)
        for p in list(seen_plates)[-10:]:
            cv2.putText(
                plate_list_img,
                p,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 35

        # Show the frame with drawings
        cv2.imshow("Video", frame)

        # Show the plate list window
        cv2.imshow("Detected Plates List", plate_list_img)

        # Save this frame to output video file
        out.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ------------------ CLEAN UP ------------------ #

    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Done! Video processed.")


# ------------------ ENTRY POINT ------------------ #

if __name__ == "__main__":
    # Folder where this script is saved
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Change these if your file names are different
    video_path = os.path.join(BASE_DIR, "car.mp4")
    model_path = os.path.join(BASE_DIR, "license_plate_detector.pt")
    # Example: if your model file is best.pt, use:
    # model_path = os.path.join(BASE_DIR, "best.pt")

    output_video_path = os.path.join(BASE_DIR, "output_simple.mp4")
    csv_path = os.path.join(BASE_DIR, "plates_simple.csv")

    print("Video path:", video_path)
    print("Model path:", model_path)

    process_video(video_path, model_path, output_video_path, csv_path)
