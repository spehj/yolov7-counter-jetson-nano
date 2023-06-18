# import cv2
# def gstreamer_pipeline(
# input_file,
# capture_width=3280,
# capture_height=2464,
# display_width=816,
# display_height=616,
# framerate=21,
# flip_method=2,
# ):
#   return (
#   'filesrc location="{0}" ! '
#   'decodebin ! '
#   'videoconvert ! '
#   'video/x-raw, width=(int){1}, height=(int){2}, format=(string)BGR ! '
#   'videoconvert ! '
#   'appsink'
#   .format(
#   input_file,
#   display_width,
#   display_height
#   )
#   )

# # Example usage with an MP4 file
# input_file_path = 'file_example_MP4_480_1_5MG.mp4'
# pipeline = gstreamer_pipeline(input_file_path)

# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


import cv2

def gstreamer_pipeline(
input_file,
capture_width=3280,
capture_height=2464,
display_width=816,
display_height=616,
framerate=21,
flip_method=2,
):
  return (
  'filesrc location=output1.mp4 ! '
  'decodebin ! '
  'videoconvert ! '
  'video/x-raw, width=(int){1}, height=(int){2}, format=(string)BGR ! '
  'videoconvert ! '
  'appsink'
  .format(
  input_file,
  display_width,
  display_height
  )
  )

# Example usage with an MP4 file
input_file_path = 'outpu1.mp4'
pipeline = gstreamer_pipeline(input_file_path)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)




