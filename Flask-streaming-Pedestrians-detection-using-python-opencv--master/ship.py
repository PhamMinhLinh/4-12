from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
import sys
import cv2
import numpy as np
import threading
import queue
import os
import tensorflow as tf
import argparse
from flask import Flask, render_template, Response





# Loai Camera cổng usb
# camera_type = 'usb'
# parser = argparse.ArgumentParser()
# parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
#                     action='store_true')
# args = parser.parse_args()

# set path cho thu muc lam viec
sys.path.append('...')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# sử dụng module ssdlite làm mạng noron cơ sở
MODEL_NAME = 'model'

# Lấy đường dẫn đến thư mục làm việc hiện tại
CWD_PATH = os.getcwd()

# Đường dẫn đến tệp .pb chứa mô hình được sử dụng để phát hiện đối tượng.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, '100K-steps.pb')

# Đường dẫn đến tệp nhãn
PATH_TO_LABELS = os.path.join(CWD_PATH, 'model', 'label_map.pbtxt')

# Số lớp mà trình phát hiện đối tượng có thể xác định
NUM_CLASSES = 3

## Tải bản đồ nhãn.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Tải mô hình Tensorflow vào bộ nhớ.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Xác định các tensor đầu vào và đầu ra (tức là dữ liệu) cho trình phân loại phát hiện đối tượng
# Tensor đầu vào là hình ảnh
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Tensor đầu ra là các hộp phát hiện, điểm số và các lớp
# Mỗi hộp đại diện cho một phần của hình ảnh nơi phát hiện một đối tượng cụ thể
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
i = detection_boxes
# Mỗi điểm thể hiện mức độ tự tin cho từng đối tượng.
# Điểm được hiển thị trên hình ảnh kết quả, cùng với nhãn lớp.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Số lượng đối tượng được phát hiện
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Khởi tạo tính toán tốc độ khung hình
frame_rate_calc = 30
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = queue.Queue()
#
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
#
def gen():

 def grab(cam, queue, width, height, fps):
    global running

    capture = cv2.VideoCapture(cam)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    while (running):
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img
        if queue.qsize() < 1:
            queue.put(frame)


 class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()
    def reset(self):
        flag=0
        counter=0



 class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):

        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.setStyleSheet("background-color: #A9E2F3")
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)
     #   self.setCursor(Qt.BlankCursor)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)
        global running
        running = True
        capture_thread.start()


    def update_frame(self):
        if not q.empty():
            frame = q.get()
            img = frame["img"]
            img_height, img_width, img_colors = img.shape
            print(img_height)
            print(img_width)
            print(img_colors)
            #print(img.shape)
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            frame_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={ image_tensor: frame_expanded })
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                # max_boxes_to_draw=1,
                min_score_thresh=0.7)


            if category_index[classes[0][0]]['name'] == "head":

                box = np.squeeze(boxes)
                for boxes in range(len(boxes)):
                    ymin = box[boxes, 0] * 480

                    if (ymin > 200):
                        cv2.putText(img, "Fall", (300, 100), font, 1, (255, 90, 90), 5)
                        print("nga")

            elif category_index[classes[0][0]]['name'] == "fall":
                cv2.putText(img, "Waning", (380, 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                print('co dau hieu nga')

            cv2.line(img, (0, 800), (1920, 800), (255, 255, 255), 4)




            cv2.putText(img,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)




            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = 3 * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)
            #cv2.imshow("tensorflow based (%d, %d)" % (width, height), image_with_box)

 ip='rtsp://admin:123456@192.168.1.2:8554/profile0'
 capture_thread = threading.Thread(target=grab, args=(0, q, 1080, 720, 30))





 app = QtWidgets.QApplication(sys.argv)
 w = MyWindowClass(None)
 w.setWindowTitle('Linh')
 w.show()
 app.exec_()
 frame = cv2.imencode('.jpg', capture_thread)[1].tobytes()
 yield (b'--frame\r\n'b'Content-Type: capture_thread/jpeg\r\n\r\n' + frame + b'\r\n')
 # time.sleep(0.1)
 key = cv2.waitKey(20)




@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





