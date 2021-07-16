import argparse
import subprocess


def parse_cmd_output(text):
    # Cut technical information from cmd output
    text_transform = text.split(b'[ INFO ]')
    result = str()
    for part_text in text_transform:
        if part_text.startswith((b' Question', b' ---answer', b' Get')):
            result += part_text.decode()
    return result


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    return parser


def generate_cmd(path_to_demo, path_to_model, sites, questions, res):
    for site_tmp in sites:
        for q in questions:
            # Prepare text command line
            cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site_tmp}" --questions "{q}"'
            # Run subprocess using prepared command line
            returned_output = subprocess.check_output(cmd)
            # Process output
            res += parse_cmd_output(returned_output)
    return res

def main():
    args = build_argparser().parse_args()

    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021.3.394/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = args.m #"bert-small-uncased-whole-word-masking-squad-0001/FP32/bert-small-uncased-whole-word-masking-squad-0001.xml"

    cmd = str()

    questions = str()
    
    with open(args.q, 'r') as file_questions:
        questions = file_questions.read().split('\n')

    sites = str()

    with open(args.i, 'r') as file_sites:
        sites = file_sites.read().split('\n')

    res = str()

    if len(sites) > 1:
        res = generate_cmd(path_to_demo, path_to_model, sites, questions, res)
    else:
        res = generate_cmd(path_to_demo, path_to_model, sites, questions, res)

    # Write output to file
    with open('answer.txt', 'w') as the_file:
        the_file.write(f'{res}\n')


if __name__ == "__main__":
    main()
