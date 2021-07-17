import argparse
import subprocess
import re


def parse_cmd_output(text):
    # Cut technical information from cmd output
    text = str(text)

    pattern = r"---answer:[^\\]*"
    info = re.findall(pattern, text)
    formated_info = [answer[11:] for answer in info]

    return formated_info


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
    path_to_model = args.m
    path_to_questions = args.q
    path_to_sites = args.i

    with open(path_to_sites, "r") as fsites:
        for index, site in enumerate(fsites):
            with open(path_to_questions, "r") as fquestions, open(f"site_{index}.txt", "w") as fsite:
                for question in fquestions:
                    cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions "{question}"'
                    returned_output = subprocess.check_output(cmd)
                    answer = parse_cmd_output(returned_output)

                    fsite.write(f'{question}\n')
                    for ans in answer:
                        fsite.write(f'{ans}\n')
                    fsite.write('\n')


if __name__ == "__main__":
    main()
