import re
import sys
import argparse

# Setup command-line arguments using argparse
parser = argparse.ArgumentParser(description='Reorder lines based on operand order in an ONNX model file.')
parser.add_argument('file_path', type=str, help='Path to the input file for reordering.')
args = parser.parse_args()

file_path = args.file_path;
cpu_operands = ['.range'];
cpu_operators_found = set();
# operators whose second parameter need to be on CPU
cpu_operators_2nd_parameter = ['.expand', '.generateConstantOfShape', '.slice', '.squeeze', '.unsqueeze', '.reshape'];
cpu_operators_3nd_parameter = ['.slice'];
cpu_operators_4th_parameter = ['.slice'];
cpu_operators_5th_parameter = ['.slice'];

def contains_word(line, word_list):
    return any(word in line for word in word_list)

with open(file_path, 'r') as file:
    lines = file.readlines()
    lines.reverse()  # Reverse the list of lines
    for line in lines:
        if "builder.constant" in line:
            continue;
        
        # extract operands in this line
        words = re.findall(r'\boperand_[a-zA-Z0-9_]+\b', line)
        if ".shape();" in line and words[0] in cpu_operands:
            # shape already returns a cpu operand, no need to 
            # track these operands.
            cpu_operands.remove(words[0]);
        if len(words) > 0 and contains_word(line, cpu_operands):
            cpu_operands.extend(words[1:])
        elif len(words) >= 2 and contains_word(line, cpu_operators_2nd_parameter):
            cpu_operands.append(words[2])
            if len(words) >= 3 and contains_word(line, cpu_operators_3nd_parameter):
                cpu_operands.append(words[3])
            if len(words) >= 4 and contains_word(line, cpu_operators_4th_parameter):
                cpu_operands.append(words[4])
            if len(words) >= 5 and contains_word(line, cpu_operators_5th_parameter):
                cpu_operands.append(words[5])
            
# Start forward pass to annotate the ops peroducing these operands as Cpu Ops
with open(file_path, 'r') as file:
    for line in file:
        words = re.findall(r'\boperand_[a-zA-Z0-9_]+\b', line)
        if len(words) > 0 and words[0] in cpu_operands:
            operator = re.findall(r'\bbuilder.(.*)\(', line)
            if (len(operator) == 1):
                cpu_operators_found.add(operator[0])
                line = line.replace(operator[0], "cpu_" + operator[0]);
        print(line, end='');

print(cpu_operators_found, file=sys.stderr);