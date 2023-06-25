import threading
import cv2
import time
from queue import Queue

class InferThread(threading.Thread):
    def __init__(self, frame_queue, max_queue_size):
        super().__init__()
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size

    def run(self):
        cap = cv2.VideoCapture('pigs-trimmed-h264-1080p.mov')
        while True:
            if self.frame_queue.qsize() < self.max_queue_size:
                ret, frame = cap.read()
                print("Reading ...")
                if not ret:
                    break
                
                # This code simulates preprocessing and inference
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_queue.put(gray_frame)  # Add frame to the queue
                time.sleep(0.1)
                #################################################
            else:
                print("SLEEPING")
                time.sleep(0.1)  # Wait if queue is full

        cap.release()


class DisplayThread(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue

    def run(self):
        while True:
            frame = self.frame_queue.get()  # Get frame from the queue
            # This code simulates postprocessing, tracking and inference
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            time.sleep(0.3)
            ############################################################
            print("Displaying ...")
            cv2.imshow('Display', blurred_frame)
            cv2.waitKey(1)
            self.frame_queue.task_done()


if __name__ == '__main__':
    max_queue_size = 10  # Maximum size of the frame queue
    frame_queue = Queue(maxsize=max_queue_size)

    infer_thread = InferThread(frame_queue, max_queue_size)
    display_thread = DisplayThread(frame_queue)

    infer_thread.start()
    display_thread.start()

    infer_thread.join()
    frame_queue.join()
    display_thread.join()

    cv2.destroyAllWindows()
