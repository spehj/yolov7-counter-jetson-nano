"""
An example that uses TensorRT's Python api to make inferences.
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
from sort1 import SortTracker

CONF_THRESH = 0.5 # was 0.5
IOU_THRESHOLD = 0.45
    
def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


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


class YoLov7TRT(object):
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

    # def infer(self, image):
    #     threading.Thread.__init__(self)
    #     # Make self the active context, pushing it on top of the context stack.
    #     start = time.time()
    #     self.ctx.push()
    #     # Restore
    #     stream = self.stream
    #     context = self.context
    #     engine = self.engine
    #     host_inputs = self.host_inputs
    #     cuda_inputs = self.cuda_inputs
    #     host_outputs = self.host_outputs
    #     cuda_outputs = self.cuda_outputs
    #     bindings = self.bindings
    #     # Do image preprocess
    #     batch_image_raw = []
    #     batch_origin_h = []
    #     batch_origin_w = []
    #     batch_input_image = np.empty(shape=[1, 3, self.input_h, self.input_w])

    #     start_pre = time.time()
    #     input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
    #     end_pre = time.time()
    #     batch_image_raw.append(image_raw)
    #     batch_origin_h.append(origin_h)
    #     batch_origin_w.append(origin_w)
    #     np.copyto(batch_input_image, input_image)
    #     batch_input_image = np.ascontiguousarray(batch_input_image)

    #     # Copy input image to host buffer
    #     np.copyto(host_inputs[0], batch_input_image.ravel())
        
    #     # Transfer input data  to the GPU.
    #     cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    #     # Run inference.
    #     context.execute_async(
    #         batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle
    #     )
    #     # Transfer predictions back from the GPU.
    #     cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    #     # Synchronize the stream
    #     stream.synchronize()
    #     end = time.time()

    #     # Remove any context from the top of the context stack, deactivating it.
    #     self.ctx.pop()
    #     # Here we use the first row of output in that batch_size = 1
    #     output = host_outputs[0]

    #     start_post = time.time()
    #     # Do postprocess
    #     result_boxes, result_scores, result_classid = self.post_process(
    #         output[0:6001], batch_origin_h[0], batch_origin_w[0]
    #     )
    #     end_post = time.time()

    #     num_of_objects = len(result_classid)

    #     # Draw rectangles and labels on the original image
    #     for j in range(len(result_boxes)):
    #         box = result_boxes[j]
    #         plot_one_box(
    #             box,
    #             image_raw,
    #             color=(50, 255, 50),
    #             label="{}:{:.2f}".format(
    #                 categories[int(result_classid[j])], result_scores[j]
    #             ),
    #         )
        
    #     return image_raw, end - start, num_of_objects, end_pre-start_pre, end_post-start_post

    def infer(self, image, tracker):
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

        start_post = time.time()
        # Do postprocess, result: [x1, y1, x2, y2, confidence, class_id]
        boxes = self.post_process_new(output, origin_h, origin_w)# , result_scores, result_classid = self.post_process_new(output, origin_h, origin_w)
        
        tracker_boxes = tracker.update(boxes)

        result_boxes, result_trackid, result_classid, result_scores = self.post_process_tracking(tracker_boxes)

        num_of_objects = 0 # len(result_classid)

        # # Draw rectangles and labels on the original image
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            plot_one_box(
                box,
                image_raw,
                color=(50, 255, 50),
                label="{}:{:.2f} id:{}".format(
                    categories[int(result_classid[j])], result_scores[j], int(result_trackid[j])
                ),
            )
        end_post = time.time()

        return image_raw, end - start, num_of_objects, end_pre - start_pre, end_post - start_post



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
    
    def post_process_tracking(self, tracking_boxes):
        result_boxes = tracking_boxes[:, :4] if len(tracking_boxes) else np.array([])
        result_trackid = tracking_boxes[:, 4] if len(tracking_boxes) else np.array([])
        result_classid = tracking_boxes[:, 5] if len(tracking_boxes) else np.array([])
        result_scores = tracking_boxes[:, 6] if len(tracking_boxes) else np.array([])
        
        return result_boxes, result_trackid, result_classid, result_scores
    
    def post_process_new(self, output, origin_h, origin_w):
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
        # result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        # result_scores = boxes[:, 4] if len(boxes) else np.array([])
        # result_classid = boxes[:, 5] if len(boxes) else np.array([])
        # 
        return boxes # , result_boxes, result_scores, result_classid

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
        # Do nms
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD
        )
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

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


class inferThread(threading.Thread):
    def __init__(self, yolov7_wrapper, video_path):
        threading.Thread.__init__(self)
        self.yolov7_wrapper = yolov7_wrapper

        width = 416
        height = 416
        framerate = 30
        # video_path = 'output1.mp4'
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # video_path = os.path.join(script_dir, video_path)
        # Pipeline working!
        gs_pipeline = "filesrc location={} ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink".format(video_path)
        # gs_pipeline = f"filesrc location=output1.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink"

        # gs_pipeline = "gst-launch-1.0 rtspsrc location=rtsp://192.168.1.5:8080/h264_ulaw.sdp latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! appsink drop=1"

        print(f"gst-launch-1.0 {gs_pipeline}\n")
        # TODO: This needs an optimization
        self.cap = cv2.VideoCapture(gs_pipeline, cv2.CAP_GSTREAMER)
        # self.cap = cv2.VideoCapture(video_path)
        # W, H = 416, 416
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FPS, 30)


        # Check if the video file was successfully loaded
        if not self.cap.isOpened():
            print("Error opening video file")

    def run(self):
        prev_frame_time = time.time()
        new_frame_time = 0
        tracker = SortTracker(max_age=3, min_hits=3, iou_threshold=0.3)
        while True:
            start_read = time.time()
            ret, frame = self.cap.read()
            end_read = time.time()

            new_frame_time = time.time()
            # Calculate FPS of complete processing time of one frame (not just inference)
            fps_total = float(1 / (new_frame_time - prev_frame_time))
            time_total = new_frame_time - prev_frame_time
            prev_frame_time = new_frame_time
            
            
            if not ret:
                break
            
            
            # img = self.image_resize(frame, width=416)
            img=frame
            start_infer = time.time()
            result, use_time, number_of_objects, time_pre, time_post = self.yolov7_wrapper.infer(img, tracker)
            end_infer = time.time()

            

            # Inference FPS:
            fps_infer = 1 / (use_time)
            fps_infer = format(fps_infer, '.2f')
            fps_str = str(fps_infer) + " FPS"
            yolo_model_name = "YOLOv7-tiny TRT"
            font = cv2.FONT_HERSHEY_DUPLEX

            start_disp = time.time()
            cv2.imshow("Recognition result", result)
            end_disp = time.time()


            print(
                "time total: {:.2f}ms, total infer: {:.2f}ms, time inference: {:.2f}ms, time pre: {:.2f}ms, time post: {:.2f}ms, diff: {:.2f}ms,  read: {:.2f}ms".format(
                    time_total*1000, (end_infer-start_infer)*1000, use_time * 1000, time_pre*1000, time_post*1000, ((end_infer-start_infer)-use_time-time_pre-time_post)*1000 , (end_read-start_read)*1000
                )
            )
            print(
                "total fps: {:.2f}, total infer fps: {:.2f}, inference fps: {:.2f},   preprocessing fps: {:.2f}ms, postprocessing fps: {:.2f}, reading fps: {:.2f}".format(
                    fps_total, 1/(end_infer-start_infer), 1 / (use_time) ,1/time_pre, 1/time_post, 1/(end_read-start_read)
                )
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
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


if __name__ == "__main__":
    # load custom plugin and engine
    # Version with input image of 416x416 pixels
    PLUGIN_LIBRARY = "libmyplugins.so"
    engine_file_path = "yolov7-tiny-rep-best.engine"
    video_path = "pigs-trimmed-h264-1080p.mov"# "output1.mp4" # file_example_MP4_480_1_5MG.mp4

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = [
        "pig",
        "person",
    ]

    if os.path.exists("output/"):
        shutil.rmtree("output/")
    os.makedirs("output/")
    # a YoLov7TRT instance
    yolov7_wrapper = YoLov7TRT(engine_file_path)
    try:
        print("batch size is", yolov7_wrapper.batch_size)

        # create a new thread to do inference
        thread1 = inferThread(yolov7_wrapper, video_path=video_path)
        thread1.start()
        thread1.join()

    finally:
        # destroy the instance
        yolov7_wrapper.destroy()
