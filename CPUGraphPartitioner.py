import re
import sys

file_path = '.\RapidChat\models\model_reordered.js'
cpu_operands = [];
cpu_operators_found = set();
# operators whose seconds parameter needs to be on CPU
cpu_operators_2nd_parameter = ['expand', 'generateConstantOfShape'];

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
        if len(words) > 0 and words[0] in cpu_operands:
            operator = re.findall(r'\bbuilder.(.*)\(', line)
            if (len(operator) > 0):
                cpu_operators_found.add(operator[0]);
            else:
                # these are typically let statements let xyx_constant = 1;
                cpu_operands.remove(words[0]);
            cpu_operands.extend(words[1:])
        elif len(words) >= 2 and contains_word(line, cpu_operators_2nd_parameter):
            cpu_operands.append(words[2])
            
# Start forward pass to annotate the ops peroducing these operands as Cpu Ops
with open(file_path, 'r') as file:
    for line in file:
        words = re.findall(r'\boperand_[a-zA-Z0-9_]+\b', line)
        if len(words) > 0 and words[0] in cpu_operands:
            operator = re.findall(r'\bbuilder.(.*)\(', line)
            if (len(operator) == 1):
                line = line.replace(operator[0], "cpu_" + operator[0]);
        print(line, end='');

print(cpu_operators_found, file=sys.stderr);