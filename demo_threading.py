import cv2
import threading
import time

def capture_frames():
    video_file = 'pigs-trimmed-h264-1080p.mov'  # Replace with the actual filename
    video_capture = cv2.VideoCapture(video_file)
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Convert frame to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the image in another thread
        display_thread = threading.Thread(target=display_frame, args=(gray_frame,))
        display_thread.start()
        
        time.sleep(0.05)  # Wait for 0.2 seconds before capturing the next frame
    
    video_capture.release()

def display_frame(frame):
    cv2.imshow('Black and White Image', frame)
    cv2.waitKey(1)  # Display the image for a short time

    # Close the window if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Start capturing frames in one thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Wait for the capture thread to finish
capture_thread.join()
