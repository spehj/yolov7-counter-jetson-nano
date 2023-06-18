from threading import Thread
import cv2, time

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        print("Video FPS: ", self.video_fps)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        processing_fps = 10
        if self.video_fps > processing_fps:
          skip_rate = round(self.video_fps/processing_fps)
        else:
          skip_rate = 1

        frame_no = 0  # Local variable to keep track of video frame number
        processed_frame_count = 0  # Local variable to count total processing frames

        total_read_time = 0

        start = time.time()
        # Read the next frame from the stream in a different thread
        while True:
            tmp = time.time()
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                total_read_time = (time.time() - tmp)
                frame_no += 1
                print("Read time: ", total_read_time)
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        self.frame = self.image_resize(self.frame, width=416)
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

if __name__ == '__main__':
    src = 'file_example_MP4_480_1_5MG.mp4'
    video_stream_widget = VideoStreamWidget(src)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass

# import cv2
# from threading import Thread
# import time
# class ThreadedCamera(object):
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
#         FPS = 1/X
#         X = desired FPS
#         self.FPS = 1/30
#         self.FPS_MS = int(self.FPS * 1000)
        
#         Start frame retrieval thread
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
        
#     def update(self):
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()
#             time.sleep(self.FPS)
            
#     def show_frame(self):
#         cv2.imshow('frame', self.frame)
#         cv2.waitKey(self.FPS_MS)

# if __name__ == '__main__':
#     src = 'file_example_MP4_480_1_5MG.mp4'
#     threaded_camera = ThreadedCamera(src)
#     while True:
#         try:
#             threaded_camera.show_frame()
#         except AttributeError:
#             pass