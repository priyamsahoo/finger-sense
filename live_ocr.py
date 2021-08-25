# Real-time Text Detection using OpenCV and Tesseract OCR

import cv2
import pytesseract
from pytesseract import Output

# Initialize the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    # Perform text detection using Tesseract OCR
    extracted_data = pytesseract.image_to_data(frame, output_type=Output.DICT)
    num_boxes = len(extracted_data['text'])

    for i in range(num_boxes):
        if int(extracted_data['conf'][i]) > 60:
            (text, x, y, w, h) = (extracted_data['text'][i], extracted_data['left'][i], extracted_data['top'][i],
                                  extracted_data['width'][i], extracted_data['height'][i])
            # Ensure the text is not empty or only whitespace
            if text and text.strip() != "":
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Display the resulting frame with text bounding boxes
    cv2.imshow('Text Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the webcam and close the display window
video_capture.release()
cv2.destroyAllWindows()
