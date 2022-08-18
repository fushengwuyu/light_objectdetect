# author: sunshine
# datetime:2022/8/17 下午2:08
"""
https://github.com/dog-qiuqiu/Yolo-FastestV2
"""
import cv2
import numpy as np
from backend import ONNXPredict, OpenvinoPredict, OpencvPredict


class YoloFastestV2:
    def __init__(self, onnx_path, names, backend='onnx', objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open(names, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array(
            [12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
            dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.IN_IMAGE_H = 352
        self.IN_IMAGE_W = 352
        self.backend = backend
        if backend == 'onnx':
            self.head = ONNXPredict(onnx_path)
        elif backend == 'openvino':
            self.head = OpenvinoPredict(onnx_path)
        elif backend == 'opencv':
            self.head = OpencvPredict(onnx_path)

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, prediction):

        outputs = np.zeros((prediction.shape[0] * self.anchor_num, 5 + len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind + j * length
                left = 4 * j
                outputs[top:top + length, 0:2] = (prediction[row_ind:row_ind + length,
                                                  left:left + 2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (prediction[row_ind:row_ind + length,
                                                  left + 2:left + 4] * 2) ** 2 * np.repeat(
                    self.anchors[i, j, :].reshape(1, -1), h * w, axis=0)
                outputs[top:top + length, 4] = prediction[row_ind:row_ind + length, 4 * self.anchor_num + j]
                outputs[top:top + length, 5:] = prediction[row_ind:row_ind + length, 5 * self.anchor_num:]
            row_ind += length

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        classIds = []
        confidences = []
        boxes = []
        for detection in outputs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence * detection[4]))
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

    def pre_processing(self, img):
        resized = cv2.resize(img, (self.IN_IMAGE_H, self.IN_IMAGE_W), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in

    def detect(self, srcimg):

        img_in = self.pre_processing(srcimg) if self.backend != 'opencv' else srcimg.copy()
        outs = self.head.do_inference(img_in)[0]
        frame = self.postprocess(srcimg, outs)

        return frame


if __name__ == '__main__':
    srcimg = cv2.imread('static/000004.jpg')
    model = YoloFastestV2(onnx_path='static/yolo-fastestv2.onnx', names='coco.names',
                          backend='openvino',
                          objThreshold=0.3,
                          confThreshold=0.3,
                          nmsThreshold=0.4)

    frame = model.detect(srcimg)

    cv2.imwrite("test_result.png", frame)
