import onnx
import argparse

# Setup command-line arguments using argparse
parser = argparse.ArgumentParser(description='Process or merge ONNX models.')
parser.add_argument('model', type=str, help='Path to the ONNX model file to load and possibly save.')
args = parser.parse_args()

# Use the provided argument instead of a hardcoded filename
model = onnx.load_model(args.model)
onnx.save(model, f'{args.model}_merged.onnx')
