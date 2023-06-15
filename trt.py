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
from queue import Queue
import threading

CONF_THRESH = 0.5 # was 0.5
IOU_THRESHOLD = 0.45
stop_processing = False

    
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
        self.categories = [
            "pig",
            "person",
        ]
        print("Engine: ", engine_file_path)
        

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            print(engine)
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

    def infer(self, image, image_queue):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[1, 3, self.input_h, self.input_w])

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle
        )
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()

        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        start_post = time.time()
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output[0:6001], batch_origin_h[0], batch_origin_w[0]
        )
        end_post = time.time()

        num_of_objects = len(result_classid)

        # Draw rectangles and labels on the original image
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            plot_one_box(
                box,
                image_raw,
                color=(50, 255, 50),
                label="{}:{:.2f}".format(
                    self.categories[int(result_classid[j])], result_scores[j]
                ),
            )
        # Put image in queue
        image_queue.put(image_raw)

        return image_raw, end - start, num_of_objects, end_post - start_post


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
            input_image_path: str, image path
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
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
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


class InferThread(threading.Thread):
    def __init__(self, yolov7_wrapper):
        threading.Thread.__init__(self)
        self.yolov7_wrapper = yolov7_wrapper

        self.cap = cv2.VideoCapture(
            "output1.mp4"
        )

        # Check if the video file was successfully loaded
        if not self.cap.isOpened():
            print("Error opening video file")

    def run(self):
        prev_frame_time = 0
        new_frame_time = 0
        total_number_of_objects = 0
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            img = cv2.resize(frame, (416, 416))

            
            result, use_time, number_of_objects = self.yolov7_wrapper.infer(img)

            total_number_of_objects += number_of_objects
            # Calculate FPS of complete processing time of one frame (not just inference)
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            # TODO
            print(
                "inference: time->{:.2f}ms, fps: {:.2f}, total fps: {:.2f}, frame obj: {}".format(
                    use_time * 1000, 1 / (use_time), fps, number_of_objects
                )
            )

           
            
            # Inference FPS:
            fps_infer = 1 / (use_time)
            fps_infer = format(fps_infer, '.2f')
            fps = str(fps_infer) + " FPS"
            yolo_model_name = "YOLOv7-tiny TRT"
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                result,
                yolo_model_name,
                (10, 50),
                font,
                1,
                (245, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                fps,
                (640 - 200, 50),
                font,
                1,
                (20, 20, 245),
                2,
                cv2.LINE_AA,
            )
            # TODO:
            # Send image to another thread where images will be displayed so this thread is not blocked. Ignore next line
            # cv2.imshow("Recognition result", result) 
            
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def display_frames(image_queue):
    prev_frame_time = 0
    prev_display_time = 0
    while True:
        
        if stop_processing:
            break
        if not image_queue.empty():
            # Calculate fps between getting new frames
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time            
            image = image_queue.get()

            # Calculate time of displaying each image
            display_time = time.time()
            image = cv2.resize(image, (416, 416))
            cv2.imshow('Video', image)
            print("Video: time-> fps: {:.2f} | display: {:.2f}".format(fps, display_time-prev_display_time))
            prev_display_time = display_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # load custom plugin and engine
#     # Version with input image of 416x416 pixels
#     PLUGIN_LIBRARY = "libmyplugins.so"
#     engine_file_path = "yolov7-tiny-rep-best.engine"

#     if len(sys.argv) > 1:
#         engine_file_path = sys.argv[1]
#     if len(sys.argv) > 2:
#         PLUGIN_LIBRARY = sys.argv[2]

#     ctypes.CDLL(PLUGIN_LIBRARY)

#     # load coco labels

#     categories = [
#         "pig",
#         "person",
#     ]

#     if os.path.exists("output/"):
#         shutil.rmtree("output/")
#     os.makedirs("output/")
#     image_queue = Queue()
#     # a YoLov7TRT instance
#     yolov7_wrapper = YoLov7TRT(engine_file_path, image_queue)
#     display_thread = threading.Thread(target=display_video, args=([],))
#     display_thread.start()

#     print("batch size is", yolov7_wrapper.batch_size)
    

#     # Create a new thread to do inference
#     thread1 = InferThread(yolov7_wrapper)
#     # Create a new thread to display results in real time

#     thread1.start()
#     thread1.join()
#     display_thread.join()


#     # destroy the instance
#     yolov7_wrapper.destroy()

def main():
    PLUGIN_LIBRARY = "libmyplugins.so"
    engine_file_path = "yolov7-tiny-rep-best.engine"
    video_path = "output1.mp4"

    ctypes.CDLL(PLUGIN_LIBRARY)
    image_queue = Queue()

    # Create an instance of YoLov7TRT
    yolo = YoLov7TRT(engine_file_path)

    # Create a thread for displaying the frames
    display_thread = threading.Thread(target=display_frames, args=(image_queue,))
    display_thread.start()

    # Read video frames
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Perform inference on the frame
        _, use_time, _,post_time = yolo.infer(frame, image_queue)
        print("inference: time->{:.2f}ms, fps: {:.2f}, postprocessing time: {}ms ".format(use_time * 1000, 1 / (use_time), post_time * 1000))

    # Set the flag to stop the video processing
    stop_processing = True

    # Wait for the display thread to finish
    display_thread.join()

    # Release the video capture
    video_capture.release()

    # Destroy the YOLO instance
    yolo.destroy()

if __name__ == '__main__':
    main()
