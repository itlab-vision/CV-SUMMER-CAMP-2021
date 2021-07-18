import argparse
import subprocess


def parse_cmd_output(text):
    # Cut technical information from cmd output

    text = text.split(b'[ INFO ]')
    answers = str()
    for line in text:
        if line.startswith((b' Question', b' ---answer', b' Get')):
            answers += line.decode()
    return answers


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
    path_to_model = args.m

    with open(args.q, 'r') as q:
        questions = q.read().split('\n')
    with open(args.i, 'r') as s:
        sites = s.read().split('\n')
    answers = str()

    # Prepare text command line 
    for site in sites:
        for question in questions:
            cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions "{question}"'

    # Run subprocess using prepared command line
    returned_output = subprocess.check_output(cmd)
    
    # Process output
    answers += parse_cmd_output(returned_output)

    # Write output to file
    with open('answers.txt', 'w') as the_file:
        the_file.write(f'{answers}\n')


if __name__ == "__main__":
    main()
