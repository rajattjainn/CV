import cv2
import numpy as np

def get_no_frame_image():
    """
    Create and returns an image to show when no frame 
    is detected in the video feed. 
    
    """
    # ToDo: Need to make the image smarter --> calculate
    # width and height of the window before creating 
    # the image
    image = 0 * np.ones((500, 500, 3), dtype="uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (00, 185)
    fontScale=3
    thickness=2
    color=(256, 256, 256)
    image = cv2.putText(image, 'No Frame Detected', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    return image

def start_webcam_capture():
    """
    Captures feed from the webcam attached to the system
    and display the frames captured unless the user hits 
    "q" on their keyboard.
    
    If no frame is captured, then it calls another function
    to retrieve an image with text "No Frame Detected", and
    display the same.
    """
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if ret:
            cv2.imshow("frame", frame)
        else:
            print ("Didn't find any frame in the input")
            cv2.imshow("frame", get_no_frame_image())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #Release the VideoCapture object
    vid.release()

    # Destroy all the windows that might be alive 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam_capture()