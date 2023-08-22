# Attention

## 导出yolov8模型需要对onnx模型输出进行修改

修改**ultralytics/nn/modules/head.py**中**Detect.forward**为

```python
def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training or self.export:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
```

## 导出onnx模型

```python
from ultralytics import YOLO

model = YOLO("./yolov8s.pt")
success = model.export(format="onnx", opset=12)
```
