import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

video_file = './output1.mp4'

pipeline = Gst.parse_launch(f'uridecodebin uri=file://{video_file} ! autovideosink')
pipeline.set_state(Gst.State.PLAYING)

bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

if msg.type == Gst.MessageType.ERROR:
    error, debug = msg.parse_error()
    print(f'Error: {error.message} ({debug})')

pipeline.set_state(Gst.State.NULL)
