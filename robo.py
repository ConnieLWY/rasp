import argparse
import threading
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np
from depthai_sdk.fps import FPSHandler
import easyocr
import time


reader = easyocr.Reader(['en'])

last_execution_time = 0
last_execution_time2 = 0
start_time = time.perf_counter()

b=[]
count = 0

fire_cascade = cv2.CascadeClassifier('cascade.xml')

minArea = 200

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

from roboflow import Roboflow
rf = Roboflow(api_key="FGBImpoSUaJ2OVdvZjVI")
project = rf.workspace().project("mplates")
model = project.version(2).model

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug
shaves = 6 if args.camera else 8


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if args.camera:
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(672, 384)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Vehicle Detection Neural Network...")
    veh_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    veh_nn.setConfidenceThreshold(0.5)
    veh_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002", shaves=shaves))
    veh_nn.input.setQueueSize(1)
    veh_nn.input.setBlocking(False)
    veh_nn_xout = pipeline.create(dai.node.XLinkOut)
    veh_nn_xout.setStreamName("veh_nn")
    veh_nn.out.link(veh_nn_xout.input)

    if args.camera:
        cam.preview.link(veh_nn.input)
    else:
        veh_xin = pipeline.create(dai.node.XLinkIn)
        veh_xin.setStreamName("veh_in")
        veh_xin.out.link(veh_nn.input)

    attr_nn = pipeline.create(dai.node.NeuralNetwork)
    attr_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039", shaves=shaves))
    attr_nn.input.setBlocking(False)
    attr_nn.input.setQueueSize(1)
    attr_xout = pipeline.create(dai.node.XLinkOut)
    attr_xout.setStreamName("attr_nn")
    attr_nn.out.link(attr_xout.input)
    attr_pass = pipeline.create(dai.node.XLinkOut)
    attr_pass.setStreamName("attr_pass")
    attr_nn.passthrough.link(attr_pass.input)
    attr_xin = pipeline.create(dai.node.XLinkIn)
    attr_xin.setStreamName("attr_in")
    attr_xin.out.link(attr_nn.input)

    print("Pipeline created.")
    return pipeline

running = True
license_detections = []
vehicle_detections = []
rec_results = []
attr_results = []
fire_results = []
frame_det_seq = 0
frame_seq_map = {}
veh_last_seq = 0
lic_last_seq = 0
decoded_text = []
fire_stacked = None

if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    width = int(cap.get(3))
    height = int(cap.get(4))
    new_width = 640
    new_height = 480
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (new_width, new_height))
    fps = FPSHandler(cap)
    cv2.startWindowThread()


def veh_thread(det_queue, attr_queue):
    global vehicle_detections, veh_last_seq

    while running:
        try:
            in_dets = det_queue.get()
            vehicle_detections = in_dets.detections

            orig_frame = frame_seq_map.get(in_dets.getSequenceNum(), None)
            if orig_frame is None:
                continue

            veh_last_seq = in_dets.getSequenceNum()

            for detection in vehicle_detections:
                bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cropped_frame = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tstamp = time.monotonic()
                img = dai.ImgFrame()
                img.setTimestamp(tstamp)
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setData(to_planar(cropped_frame, (72, 72)))
                img.setWidth(72)
                img.setHeight(72)
                attr_queue.send(img)

            fps.tick('veh')
        except RuntimeError:
            continue


def fire_thread(frame):
    fire = fire_cascade.detectMultiScale(frame, 12, 3)  # test for fire detection
    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                      2)  # highlight the area of image with fire
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        # Define the range of fire color in HSV
        lower = np.array([0, 50, 50])
        upper = np.array([10, 255, 255])

        # Create a binary image where the fire color is white and everything else is black
        mask = cv2.inRange(hsv_image, lower, upper)

        # Apply morphological operations to remove noise and fill gaps
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Calculate the percentage of white pixels in the image
        total_pixels = mask.shape[0] * mask.shape[1]
        white_pixels = np.sum(mask == 255)
        white_percentage = (white_pixels / total_pixels) * 100

        # Determine if the image has fire color based on the threshold percentage
        if white_percentage >= 15:
            cv2.putText(frame, "Fire Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Smoke Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        fps.tick('fire')

def veh():
    global count
    for detection in vehicle_detections:
        area = (detection.xmax - detection.xmin) * (detection.ymax - detection.ymin)
        if area > 0.2:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            print("Car detected")
            frame2 = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # cv2.imwrite(f'vehicle/vehicle {count}.jpg', frame2)
            count = count + 1
            pos = model.predict(frame, confidence=50, overlap=30).json()
            if pos['predictions'][0] is not None:
                x_min = int(pos['predictions'][0]['x'] - pos['predictions'][0]['width'] / 2)
                y_min = int(pos['predictions'][0]['y'] - pos['predictions'][0]['height'] / 2)
                x_max = int(pos['predictions'][0]['x'] + pos['predictions'][0]['width'] / 2)
                y_max = int(pos['predictions'][0]['y'] + pos['predictions'][0]['height'] / 2)
                # Crop the license plate from the frame using the bounding box coordinates
                license_plate = frame[y_min:y_max, x_min:x_max]
                gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                noise = cv2.medianBlur(gray, 3)
                thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                result = reader.readtext(license_plate, paragraph="False",
                                         allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                for detection in result:
                    text = detection[1]
                    print(text)


print("Starting pipeline...")
with dai.Device(create_pipeline()) as device:
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        veh_in = device.getInputQueue("veh_in")

    attr_in = device.getInputQueue("attr_in")
    veh_nn = device.getOutputQueue("veh_nn", 1, False)
    attr_nn = device.getOutputQueue("attr_nn", 1, False)
    attr_pass = device.getOutputQueue("attr_pass", 1, False)


    veh_t = threading.Thread(target=veh_thread, args=(veh_nn, attr_in))
    veh_t.start()


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        global frame_det_seq

        if args.video:
            read_correctly, frame = cap.read()
            if read_correctly:
                frame_seq_map[frame_det_seq] = frame
                frame_det_seq += 1
            return read_correctly, frame
        else:
            in_rgb = cam_out.get()
            frame = in_rgb.getCvFrame()
            frame_seq_map[in_rgb.getSequenceNum()] = frame

            return True, frame


    try:
        while should_run():
            read_correctly, frame = get_frame()

            if not read_correctly:
                break

            for map_key in list(filter(lambda item: item <= min(lic_last_seq, veh_last_seq), frame_seq_map.keys())):
                del frame_seq_map[map_key]

            fps.nextIter()

            if not args.camera:
                tstamp = time.monotonic()
                veh_frame = dai.ImgFrame()
                veh_frame.setData(to_planar(frame, (300, 300)))
                veh_frame.setTimestamp(tstamp)
                veh_frame.setSequenceNum(frame_det_seq)
                veh_frame.setType(dai.RawImgFrame.Type.BGR888p)
                veh_frame.setWidth(300)
                veh_frame.setHeight(300)
                veh_frame.setData(to_planar(frame, (672, 384)))
                veh_frame.setWidth(672)
                veh_frame.setHeight(384)
                veh_in.send(veh_frame)

            if debug:
                # Check if it's been one hour since the last execution
                veh()
                fire_thread(frame)
                    
                # Exit if 'q' is pressed
                # cv2.imshow("rgb", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print("Elapsed time: ", elapsed_time)
                    break

        running = False

        veh_t.join()

    except KeyboardInterrupt or cv2.waitKey(1) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time)
        pass

cap.release()
cv2.destroyAllWindows()
