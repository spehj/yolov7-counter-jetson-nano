import cv2
import time
from multiprocessing import Process, Queue

class InferProcess(Process):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue

    def run(self):
        cap = cv2.VideoCapture('pigs-trimmed-h264-1080p.mov')
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_queue.put(gray_frame)  # Add frame to the queue
            time.sleep(0.08)
            print("INFER")

        cap.release()


class DisplayProcess(Process):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue

    def run(self):
        while True:
            frame = self.frame_queue.get()  # Get frame from the queue
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            time.sleep(0.06)
            print("DISPLAY")
            cv2.imshow('Display', blurred_frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    frame_queue = Queue()

    infer_process = InferProcess(frame_queue)
    display_process = DisplayProcess(frame_queue)

    infer_process.start()
    display_process.start()

    infer_process.join()
    display_process.join()

    cv2.destroyAllWindows()
