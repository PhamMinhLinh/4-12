import os
import cv2
import numpy as np
import tensorflow as tf
import sys
# Điều này là cần thiết vì sổ ghi chép được lưu trữ trong thư mục object_detection.

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util # tác vụ nhỏ xử lý cơ bản

MODEL_NAME = 'model'
IMAGE_NAME = 'test anh/fall/fall_img_98.jpg'
rs='anh/fall/98.jpg'


b='frozen_inference_graph2k.pb'
a='100K-steps.pb'


CWD_PATH = os.getcwd() # lấy đường dẫn ngay tại nơi mình làm việc
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, b) # nối đường dẫn hiện tại đến file 100k


PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'label_map.pbtxt')
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME) # đường dẫn tới hình

NUM_CLASSES = 3
# Tải bản đồ nhãn.
# Nhãn bản đồ các chỉ số ánh xạ đến tên danh mục, để khi tích chập của chúng tôi
# mạng dự đoán `3`
# Ở đây chúng tôi sử dụng các hàm tiện ích nội bộ, nhưng bất kỳ thứ gì trả về
# từ điển ánh xạ các số nguyên sang các nhãn chuỗi thích hợp sẽ ổn
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Nạp mô hình Tensorflow vào bộ nhớ.

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)
# xử lý
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  #nó chính là bộ lọc chạy trên hình ảnh của mình
# bài toán OD vẽ từng box sau đó classfi từng box đó thuộc lớp nào trong 3 lớp thì độ chính xác cao
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') # sẽ vẽ box lên hình
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')# là độ chính xác của box
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') # dự đoán cái box đó nằm trong class nào
num_detections = detection_graph.get_tensor_by_name('num_detections:0') # có bao nhiu dự đoán trong 1 hình

image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0) # đưa hình ảnh về ma trạn np

# dugf model của mình để có box , scor, classes, num
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})


vis_util.visualize_boxes_and_labels_on_image_array(
    image, # đầu tiên đưa nguồn muốn vẽ vào
    np.squeeze(boxes), # vẽ box
    np.squeeze(classes).astype(np.int32), # in classes ra
    np.squeeze(scores),# in độ chính xác ra,
    category_index,#in thử thuộc class nào
    use_normalized_coordinates=True,
    line_thickness=8) # dộ dau box
res = cv2.resize(image, dsize=(1240,820), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Object detector', res) # show hình lên
cv2.imwrite(rs, res) # lưu lại hình đó
cv2.waitKey(0)
cv2.destroyAllWindows()
