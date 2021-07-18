import argparse
import subprocess


def parse_cmd_output(text):
    # Cut technical information from cmd output
    info_strings = text.split(b'[ INFO ]')
    result = str()
    for s in info_strings:
        if s.startswith(b' ---answer'):
            result += s.decode()
    return result


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    return parser


def main():
    args = build_argparser().parse_args()

    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021.4.582/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = "intel/bert-small-uncased-whole-word-masking-squad-0001/FP32/bert-small-uncased-whole-word-masking-squad-0001.xml"
    # question = "What operating system is required?"
    # site = "https://en.wikipedia.org/wiki/OpenVINO"

    answers_log = list()

    with open(args.q, 'r') as file_q:
        questions = file_q.read().split('\n')

    with open(args.i, 'r') as file_s:
        sites = file_s.read().split('\n')

    for site in sites:
        answers_log.append("[ SITE ]:  " + site)
        for question in questions:
            # Prepare text command line
            cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions "{question}"'

            # Run subprocess using prepared command line
            returned_output = subprocess.check_output(cmd)

            # Process output
            answer = parse_cmd_output(returned_output)

            answers_log.append(question + "\n" + answer)

    # Write output to file
    with open('answer.txt', 'w') as ans_file:
        for answer in answers_log:
            ans_file.write(f'{answer}\n')


if __name__ == "__main__":
    main()
