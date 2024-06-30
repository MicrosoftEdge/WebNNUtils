import re
import sys
import argparse

# Setup command-line arguments using argparse
parser = argparse.ArgumentParser(description='Reorder lines based on operand order in an ONNX model file.')
parser.add_argument('file_path', type=str, help='Path to the input file for reordering.')
args = parser.parse_args()

initialized_operands = set()
def is_line_ready(line):
    operand_words = []  # Initialize an empty list to store operand words
    # Use regex to find words starting with "operand_" (ignoring symbols)
    words = re.findall(r'\boperand_[a-zA-Z0-9_]+\b', line)
    operand_words.extend(words)
    if len(operand_words) > 0:
        initialized_operands.add(operand_words[0]);
        if len(operand_words) > 1:
            is_ready = True;
            for operand in operand_words:
                if (operand not in initialized_operands):
                    is_ready = False;
            if is_ready:
                return True;
            else:
                return False;
        else:
            return True;
    else:
        return True;

# This algorithm is terrible and n^2 but gets the job done for the 
# purposes of prototyping.
unready_lines = set();
def reorder_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if is_line_ready(line):
                print(line, end='');
                for unready_line in unready_lines.copy():
                    if is_line_ready(unready_line):
                       print(unready_line, end='');
                       unready_lines.remove(unready_line);
            else:
                unready_lines.add(line);

reorder_lines(args.file_path)
assert(len(unready_lines) == 0);