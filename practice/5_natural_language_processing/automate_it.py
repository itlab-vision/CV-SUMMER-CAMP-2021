
import argparse
import subprocess

def parse_cmd_output(text):
    # Cut technical information from cmd output
    srch_str = b'---answer'
    bias = len(srch_str)
    res_str = b''
    
    cur_srch = text.find(srch_str)
    while not cur_srch == -1 :
        end_index = text.find(b'\r', cur_srch + bias )
        res_str += text[cur_srch : end_index] + b' \n'
        cur_srch = text.find(srch_str, end_index)  
        print(res_str)
    
    return res_str.decode('utf-8')


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    parser.add_argument("--fq", required=False, help="Optional. If --fq = True then program will read questions from file 'questions.txt'", default=False)
    parser.add_argument("--fs", required=False, help="Optional. If --fs = True then program will read sites from file 'pages.txt'", default=False)
    return parser


def main():
    args = build_argparser().parse_args()
    
    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = args.m 
    
    if args.fs:
        with open('pages.txt', 'r') as sites_file:
            sites = [line.replace('\n','') for line in sites_file]
    else:
        sites = [args.i]
    
    if args.fq:
        with open('questions.txt', 'r') as questions_file:
            questions = [line.replace('\n','') for line in questions_file]
    else:
        questions = [args.q]
        
    for site in sites:
    
        for question in questions:
    
            # Prepare text command line 
            cmd = f'python "{path_to_demo}" -v "vocab.txt" -m "{path_to_model}" --input="{site}" --questions "{question}"'

            # Run subprocess using prepared command line
            returned_output = subprocess.check_output(cmd)

            # Process output
            answer = parse_cmd_output(returned_output)

            # Write output to file
            with open('answer1.txt', 'a') as the_file:
                the_file.write(f'Site: {site}\n{question}\n{answer}\n')

if __name__ == "__main__":
    main()
