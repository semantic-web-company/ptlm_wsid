import os.path as op

from utils import config


evaluation_output_directory = config['evaluation']['outputfolder']
language = config['wikiner']['language']

evaluation_csv_file = op.join(evaluation_output_directory, language+"_results.csv")

with open(evaluation_csv_file) as fin:
    curr_row = 0
    header_trans = {}
    data = []
    for row in fin:
        curr_row+=1
        row_s = row.strip().split("\t")
        if curr_row==1:
            for rown,head in enumerate(row_s):
                break

