import os
with open(os.path.join("visual_results") + "/input_file.txt2.pdb") as f:
    line = f.readline()
    while line:
        print(line.strip())