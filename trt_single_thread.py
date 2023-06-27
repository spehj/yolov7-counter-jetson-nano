"""
An example that uses TensorRT's Python api to make inferences. Clean version, without threading
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from sort import SortTracker

CONF_THRESH = 0.5 # was 0.5
IOU_THRESHOLD = 0.45


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov7 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class Tracking():
    def __init__(self) -> None:
        self.counter_to_right = 0
        self.counter_to_left = 0
        self.detections = {}
    
    def process_for_tracking(self, tracking_boxes):
        """
        param:
            tracking_boxes: [x1,y1,x2,y2 track_id, class_id, conf]
        return:
            result_boxes: [[x1,y1,x2,y2] [x1,y1,x2,y2] [x1,y1,x2,y2] ]
            result_trackid: [track_id track_id track_id]
            result_classid: [classid classid classid]
            result_scores: [scores scores scores]
        """
        result_boxes = tracking_boxes[:, :4] if len(tracking_boxes) else np.array([])
        result_trackid = tracking_boxes[:, 4] if len(tracking_boxes) else np.array([])
        result_classid = tracking_boxes[:, 5] if len(tracking_boxes) else np.array([])
        result_scores = tracking_boxes[:, 6] if len(tracking_boxes) else np.array([])
        
        return result_boxes, result_trackid, result_classid, result_scores
    
    def count(self, image_raw, result_boxes, result_trackid, result_classid):
        # To do counting we need
        # Calculate center_x and center_y
        img_height, img_width = image_raw.shape[:2]
        x = img_width // 2
        counting_class = 0
        if len(result_boxes) > 0:
            center_x = (result_boxes[:, 0] + result_boxes[:, 2]) / 2
            center_y = (result_boxes[:, 1] + result_boxes[:, 3]) / 2
            # 1. Current status as [center_x, center_y, track_id, class_id]
            # Combine the arrays
            current_status = np.column_stack((center_x, center_y, result_trackid, result_classid))
            # current_status =
            # [[111.37606153 584.85730603   4.           0.        ]
            # [ 277.78122332 700.1416417    3.           0.        ]
            # [ 315.21315656 819.75725727   2.           0.        ]
            # [ 674.30197419 760.30353343   1.           0.        ]]

            # 2. A dict: 
            # self.detections = {track_id_1: [last_center_x, last_center_y, center_x, center_y, class_id], track_id_2: [last_center_x, last_center_y, center_x, center_y, class_id],}
            
            for element in current_status:
                track_id = element[2]
                if track_id in self.detections:
                    # If tracking id is present in detections, save last x,y position
                    last_x, last_y = self.detections[track_id][2], self.detections[track_id][3]
                    # Update detections dictionary - add new values for center x and y
                    self.detections[track_id] = [last_x, last_y, element[0], element[1], self.detections[track_id][4] ]
                    if self.detections[track_id][0] < x and self.detections[track_id][2] >= x and int(self.detections[track_id][4]) == counting_class:
                        self.counter_to_right +=1
                    elif self.detections[track_id][0] > x and self.detections[track_id][2] <= x and int(self.detections[track_id][4]) == counting_class:
                        self.counter_to_right -=1
                else:
                    # If tracking id is detected first time
                    last_x, last_y = None, None
                    self.detections[track_id] = [last_x, last_y, element[0], element[1], element[3]]

    def draw_counter(self, image_raw, result_boxes):
        # Draw points and labels on the original image
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            c_x, c_y = self.calculate_center(bbox=box)
            center = (c_x, c_y)
            color = (0, 0, 255) # BGR
            radius = 5
            cv2.circle(image_raw, center, radius, color, -1)
        image_raw = self.display_counter(image_raw)
        return image_raw

    def calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y

    def display_counter(self, img):
        # Draw a counter
        # Define the position of the text
        img_height, img_width = img.shape[:2]
        x = img_width // 2
        text_position = (x + 10, 50)  # Adjust the coordinates as per your requirement

        # Define the font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (255, 0, 0)  # White color in BGR format
        font_thickness = 2
        text = str(self.counter_to_right)

        # Put the text on the image
        cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)
        
        # Display line
        line_color = (0, 255, 255)  # Red color in BGR format
        line_thickness = 2
        cv2.line(img, (x, 0), (x, img_height), line_color, line_thickness)
        return img

class YoLov7TRT():
    """
    description: A YOLOv7 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print("bingding:", binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, image):
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()

        # Do image preprocess
        start_pre = time.time()
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
        end_pre = time.time()
        start = time.time()
        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], input_image.ravel())
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0])

        # Run inference.
        self.context.execute_async(
            batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0])

        # Synchronize the stream
        self.stream.synchronize()
        
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

        # Here we use the first row of output in that batch_size = 1
        output = self.host_outputs[0]
        end = time.time()
        # Do postprocess, result: [x1, y1, x2, y2, confidence, class_id]
        return output, end - start, origin_h, origin_w

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)
    
    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
        param:
            raw_bgr_image: numpy.ndarray, BGR image
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        
        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
    
    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms, Result: [x1, y1, x2, y2, confidence, class_id]
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD
        )
        return boxes # , result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, 0, None
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(
        self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4
    ):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = (
                self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            )
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class InferThread(threading.Thread):
    def __init__(self, yolov7, video_path):
        threading.Thread.__init__(self)
        self.yolov7 = yolov7
        gs_pipeline = "filesrc location={} ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink".format(video_path)
        self.cap = cv2.VideoCapture(gs_pipeline, cv2.CAP_GSTREAMER)

        # Check if the video file was successfully loaded
        if not self.cap.isOpened():
            print("Error opening video file")

    def run(self):
        tracker = SortTracker(max_age=3, min_hits=3, iou_threshold=0.3)
        tracking = Tracking()

        while True:
            ret, image_raw = self.cap.read()
            
            if not ret:
                break
 
            # Do inference
            output, use_time, origin_h, origin_w = self.yolov7.infer(image_raw)
            # Postprocessing
            boxes = self.yolov7.post_process(output, origin_h, origin_w)
            
            # Tracking
            tracker_boxes = tracker.update(boxes)
            result_boxes, result_trackid, result_classid, result_scores = tracking.process_for_tracking(tracking_boxes=tracker_boxes)
            tracking.count(image_raw=image_raw, result_boxes=result_boxes, result_trackid=result_trackid, result_classid=result_classid)
            result = tracking.draw_counter(image_raw=image_raw, result_boxes=result_boxes)

            cv2.imshow("Recognition result", result)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
  

if __name__ == "__main__":
    # load custom plugin and engine
    # Version with input image of 416x416 pixels
    PLUGIN_LIBRARY = "libmyplugins.so"
    engine_file_path = "yolov7-tiny-rep-best.engine"
    video_path = "demo.mp4"# "pigs-trimmed-2.mov"#   pigs-trimmed-h264-1080p.mov "output1.mp4" # file_example_MP4_480_1_5MG.mp4

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # Custom trained labels
    categories = [
        "pig",
        "person",
    ]

    
    # a YoLov7TRT instance
    yolov7 = YoLov7TRT(engine_file_path)
    try:
        # create a new thread to do inference
        thread1 = InferThread(yolov7, video_path=video_path)
        thread1.start()
        thread1.join()

    finally:
        # destroy the instance
        yolov7.destroy()
