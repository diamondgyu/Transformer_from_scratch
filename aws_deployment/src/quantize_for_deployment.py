from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = '../models/model-pretrained.onnx'
model_int8 = '../models/model-quantized.onnx'

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,     
    weight_type=QuantType.QInt8   
)

print(f"finished on: {model_int8}")