import numpy as np
import librosa
import tflite_runtime.interpreter as tflite

def extract_features(file, n_mfcc=13, duration=30):
    y, sr = librosa.load(file, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).astype(np.float32).reshape(1, -1)

def load_tflite_model(model_path='model_reduced.tflite'):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, features):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), output
