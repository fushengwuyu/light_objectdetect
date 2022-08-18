# author: sunshine
# datetime:2022/8/17 下午2:53
import onnxruntime
import cv2
from openvino.runtime import Core


class ONNXPredict:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)

    def do_inference(self, img_in):
        # Compute
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_in})
        return outputs


class OpencvPredict:
    def __init__(self, onnx_path, w, h):
        self.net = cv2.dnn.readNet(onnx_path)
        self.w = w
        self.h = h

    def do_inference(self, img_in):
        blob = cv2.dnn.blobFromImage(img_in, 1 / 255.0, (self.w, self.h))
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outs


class OpenvinoPredict:
    def __init__(self, engine_path):
        ie = Core()
        model = ie.read_model(engine_path)
        self.compiled_model = ie.compile_model(model=model, device_name='CPU')

    def do_inference(self, img_in):
        result_infer = self.compiled_model([img_in])
        return list(result_infer.values())
