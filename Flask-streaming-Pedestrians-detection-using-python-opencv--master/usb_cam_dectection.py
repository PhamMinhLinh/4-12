import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util

PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'model/label_map.pbtxt'

NUM_CLASSES = 3
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

font = cv2.FONT_HERSHEY_SIMPLEX

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))
        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage:%s (cameraID | filename) Detect faces\
 in the video example:%s 0" % (sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        camID = int(sys.argv[1])
    except:
        camID = sys.argv[1]

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)


    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break
        [h, w] = image.shape[:2]
        print(h, w)
        image = cv2.flip(image, 1)
        height, width, channels = image.shape
        centerX, centerY = int(height / 2), int(width / 2)
        radiusX, radiusY = int(50 * height / 100), int(50 * width / 100)
        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height))
        (boxes, scores, classes, num_detections) = tDetector.run(image)
        image_with_box = cv2.rectangle(image, (120, 40), (520, 440), (255, 255, 0), 4)
        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False
        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image_with_box)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    #python usb_cam_dectection.py 0
