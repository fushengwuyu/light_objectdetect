# author: sunshine
# datetime:2022/8/17 下午4:45

"""
https://github.com/dog-qiuqiu/FastestDet
"""
import cv2
import numpy as np
from backend import ONNXPredict, OpenvinoPredict, OpencvPredict


class FastestDet:
    def __init__(self, onnx_path, names, backend='onnx', confThreshold=0.3, nmsThreshold=0.4):

        self.classes = list(map(lambda x: x.strip(), open(names, 'r').readlines()))
        self.backend = backend
        self.inpWidth = 512
        self.inpHeight = 512
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.H, self.W = 32, 32
        self.grid = self._make_grid(self.W, self.H)

        if backend == 'onnx':
            self.head = ONNXPredict(onnx_path)
        elif backend == 'openvino':
            self.head = OpenvinoPredict(onnx_path)
        elif backend == 'opencv':
            self.head = OpencvPredict(onnx_path, self.inpWidth, self.inpHeight)

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def pre_processing(self, img):
        img_in = cv2.resize(img, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold).flatten()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def detect(self, srcimg):

        img_in = self.pre_processing(srcimg) if self.backend != 'opencv' else srcimg.copy()
        # img_in = self.pre_processing(srcimg)
        pred = self.head.do_inference(img_in)[0]

        pred[:, 3:5] = self.sigmoid(pred[:, 3:5])  ###w,h
        pred[:, 1:3] = (np.tanh(pred[:, 1:3]) + self.grid) / np.tile(np.array([self.W, self.H]),
                                                                     (pred.shape[0], 1))  ###cx,cy
        frame = self.postprocess(srcimg, pred)

        return frame


if __name__ == '__main__':
    srcimg = cv2.imread('static/000004.jpg')
    model = FastestDet(onnx_path='static/FastestDet.onnx', names='static/coco.names',
                       backend='opencv',
                       confThreshold=0.3,
                       nmsThreshold=0.4)

    frame = model.detect(srcimg)

    cv2.imwrite("static/test_result.png", frame)
