
import argparse
import subprocess
#import sys
#from pathlib import Path

#print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
#print("PATH:", os.environ.get('PATH'))
#sys.path.append('C:\Users\Владислав\Documents\Python Scripts\CV-SUMMER-CAMP-2021\practice\openvino-virtual-environments\openvinoenv\Lib\site-packages')
#sys.path.append(str(Path(__file__).resolve().parents[1] / 'openvino-virtual-environments_2/openvinoenv/Lib/site-packages'))

def parse_cmd_output(text):
    # Cut technical information from cmd output
    print(text)
    
    srch_str = b'---answer'
    bias = len(srch_str)
    res_str = b''
    
    cur_srch = text.find(srch_str)
    while not cur_srch == -1 :
        end_index = text.find(b'\r', cur_srch + bias )
        res_str += text[cur_srch : end_index] + b' \n'
        cur_srch = text.find(srch_str, end_index)  
        print(res_str)
    
    print(str(res_str))
    return res_str.decode('utf-8')


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    return parser


def main():
    args = build_argparser().parse_args()

    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = args.m # "C:\Users\Владислав\Documents\Python Scripts\CV-SUMMER-CAMP-2021\practice\5_natural_language_processing\intel\bert-small-uncased-whole-word-masking-squad-0001\FP16/bert-small-uncased-whole-word-masking-squad-0001/FP32/bert-small-uncased-whole-word-masking-squad-0001.xml"
    question = args.q #"What operating system is required?"
    site = args.i #"https://en.wikipedia.org/wiki/OpenVINO"

    # Prepare text command line 
    cmd = f'python "{path_to_demo}" -v "vocab.txt" -m "{path_to_model}" --input="{site}" --questions "{question}"'

    # Run subprocess using prepared command line
    returned_output = subprocess.check_output(cmd)

    # Process output
    answer = parse_cmd_output(returned_output)

    # Write output to file
    with open('answer.txt', 'a') as the_file:
        the_file.write(f'{question}\n{answer}\n\n')

if __name__ == "__main__":
    
    main()
