import argparse
import subprocess


def parse_cmd_output(text):
    # Cut technical information from cmd output

    return text


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
    path_to_model = args.m #"bert-small-uncased-whole-word-masking-squad-0001/FP32/bert-small-uncased-whole-word-masking-squad-0001.xml"
    question = "What operating system is required?"
    site = "https://en.wikipedia.org/wiki/OpenVINO"

    # Prepare text command line 
    cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions "{question}"'

    # Run subprocess using prepared command line
    returned_output = subprocess.check_output(cmd)

    # Process output
    answer = parse_cmd_output(returned_output)

    # Write output to file
    with open('answer.txt', 'w') as the_file:
        the_file.write(f'{answer}\n')


if __name__ == "__main__":
    main()
