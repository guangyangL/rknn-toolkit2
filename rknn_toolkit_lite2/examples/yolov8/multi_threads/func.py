#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time
 
QUANTIZE_ON = True
 
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640
 
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

def xywh2xyxy(x: np.ndarray):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def make_anchors(feats: np.ndarray, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(stop=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = np.arange(stop=h, dtype=dtype) + grid_cell_offset  # shift y
        sx, sy = np.meshgrid(sx, sy)
        anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=dtype))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox(distance: np.ndarray, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.array_split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox


def softmax(x, axis=-1):
    # 计算指数
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 计算分母
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # 计算 softmax
    softmax_x = exp_x / sum_exp_x
    return softmax_x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dfl(x: np.ndarray):
    c1 = 16
    b, c, a = x.shape
    conv = np.arange(0, c1, dtype=np.float32)
    conv = conv.reshape(1, 16, 1, 1)

    softmax_x = softmax(x.reshape(b, 4, c1, a).transpose(0, 2, 1, 3), 1)
    return np.sum(softmax_x * conv, 1, keepdims=True).reshape(b, 4, a)


def yolov8_head(x, anchors, nc):  # prediction head
    strides = [8, 16, 32]  # P3, P4, P5 strides
    shape = x[0].shape
    reg_max = 16
    no = nc + reg_max * 4  # number of outputs per anchor
    anchors, strides = (x.transpose(1, 0) for x in make_anchors(x, strides, 0.5))
    x_cat = np.concatenate([xi.reshape(shape[0], no, -1) for xi in x], 2)
    box, cls = np.split(x_cat, (reg_max * 4,), 1)
    dbox = dist2bbox(dfl(box), anchors[np.newaxis, :], xywh=True, dim=1) * strides
    y = np.concatenate((dbox, sigmoid(cls)), 1)

    return y


def yolov8_postprocess(
    prediction: np.ndarray,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index

    xc = np.amax(prediction[:, 4:mi], 1) > conf_thres  # scores per image

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(1, 0)[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(
            x,
            (
                4,
                nc + 4,
            ),
            1,
        )
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1
            )
        else:  # best class only
            conf = cls.max(1, keepdims=True)
            j = np.argmax(cls, 1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), 1)[
                np.squeeze(conf > conf_thres)
            ]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort()[::-1][:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (
            0 if agnostic else max_wh
        )  # classes，如果是agnostic，那么就是0，否则就是max_wh，为了对每种类别的框进行NMS
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_thres, iou_thres
        ).flatten()

        i = i[:max_det]  # limit detections

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def preprocess(img, target_size=(IMG_SIZE, IMG_SIZE)):
    h, w, c = img.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    nw, nh = int(scale * w), int(scale * h)
    img_resized = cv2.resize(img, (nw, nh))
    img_paded = np.full(shape=[target_size[1], target_size[0], c], fill_value=0).astype(
        np.uint8
    )
    dw, dh = (target_size[0] - nw) // 2, (target_size[1] - nh) // 2
    img_paded[dh : nh + dh, dw : nw + dw, :] = img_resized
    return img_paded
 

 
def myFunc(rknn_lite, IMG):
    image = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    image = preprocess(image)
    outputs = rknn_lite.inference(inputs=[image])
 
    output_head = yolov8_head(outputs, None, 80)
        # print(outputs.shape)
    output = yolov8_postprocess(
            output_head, conf_thres=0.25, iou_thres=0.45, max_det=300
        )
    
 
    result = output[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box in result:
        box = box.tolist()
        cls = int(box[5])

        box[0:4] = [int(i) for i in box[0:4]]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        score = box[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            "{} {:.2f}".format(CLASSES[cls], score),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (153, 255, 255),
            2,
        )
    return image