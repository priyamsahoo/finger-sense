import cv2
import imutils
import numpy as np
import pandas as pd
from imutils.video import VideoStream
import easyocr
import mediapipe


def main():
    reader = easyocr.Reader(['en'], gpu=False)
    handsModule = mediapipe.solutions.hands
    drawingModule = mediapipe.solutions.drawing_utils

    # Defining the video stream on capture card index 0 by default
    print("[STATUS] Starting video stream ...")
    vs = VideoStream(src=0).start()

    # Scale for down-scaling the image when processing
    # new_w, new_h = 320, 230
    new_w, new_h = 640, 480

    print("[INFO] Press q in the video feed to exit ...")
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        while True:
            # Getting the current frame, standardizing it and saving a copy
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            orig = frame.copy()

            # Downsizing the image and defining a ratio to the original size
            h, w = frame.shape[:2]
            ratio_w = w / float(new_w)
            ratio_h = h / float(new_h)
            frame = cv2.resize(frame, (new_w, new_h))

            # Parsing the frame through the text recognition and saving the results in a DataFrame
            results = reader.readtext(np.array(frame))
            df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])

            # Looping over the results from the current frame and displaying it in a box overlay
            for _, row in df.iterrows():
                box_tuples = [(int(x * ratio_w), int(y * ratio_h))
                              for x, y in row['bbox']]
                cv2.rectangle(
                    orig, box_tuples[0], box_tuples[2], (0, 255, 0), 2)
                text = f"{row['text']} - {round(row['conf'], 2)}"
                cv2.putText(orig, text, (box_tuples[3][0], box_tuples[3]
                            [1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Produces the hand framework overlay on top of the hand
            frame1 = cv2.resize(orig, (640, 480))
            # frame1 = cv2.resize(frame, (320, 230))

            results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks is not None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(
                        frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                    # Below is Added Code to find and print to the shell the Location X-Y coordinates of Index Finger, Uncomment if desired
                    for point in handsModule.HandLandmark:

                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y, 640, 480)
                        # normalizedLandmark.x, normalizedLandmark.y, 320, 230)

                        # Using the Finger Joint Identification Image we know that point 8 represents the tip of the Index Finger
                        if point == 8:
                            print(point)
                            print(pixelCoordinatesLandmark)
                            print(normalizedLandmark)

            # Show the frame with the added OCR bounding boxes and hand framework
            cv2.imshow("Live text recognition and finger tracking", frame1)

            # Check for 'q' keypress to stop the program
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Stop the resources and closing all windows
    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
