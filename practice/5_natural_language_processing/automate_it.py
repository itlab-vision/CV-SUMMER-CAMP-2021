import argparse
import subprocess
import logging as log
import sys


def parse_cmd_output(text):
    # Cut technical information from cmd output
    text_chunks = text.split(b'[ INFO ]')
    answers = str()
    old_chunk = bytes()
    for chunk in text_chunks:
        if chunk.startswith((b' Question', b' ---answer', b' Get')) or old_chunk.startswith(b' ---answer'):
            answers += chunk.decode()
            old_chunk = chunk[:11]

    log.info(answers)
    return answers


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    return parser


def _prepare_multisearch(subsites, path_to_demo, path_to_model, question_str):
    st_str = str()
    for st in subsites:
        if st != str():
            st_str += f'--input="{st}" '
    cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} {st_str} --questions {question_str}'

    return cmd


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()

    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021.4.582/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = args.m
    # "bert-small-uncased-whole-word-masking-squad-0001/FP32/bert-small-uncased-whole-word-masking-squad-0001.xml"

    # Prepare questions
    with open(args.q, 'r') as f:
        questions = f.read().split('\n')  # "What operating system is required?"

    question_str = str()
    for question in questions:
        if question != str():
            question_str += f'"{question}"' + ' '
    question_str = question_str.strip()

    # Prepare sites
    with open(args.i, 'r') as f:
        sites = f.read().split('\n')  # "https://en.wikipedia.org/wiki/OpenVINO"

    started_write = False
    for site in sites:
        if site != str():
            subsites_comma_list = [st.strip(' ') for st in site.split(',')]
            subsites_space_list = [st.strip(',') for st in site.split(' ')]
            all_subsites = [subsites_comma_list, subsites_space_list]

            # Prepare text command line
            for subsite_list in all_subsites:
                if len(subsite_list) > 1:
                    cmd = _prepare_multisearch(subsite_list, path_to_demo, path_to_model, question_str)
                    break
            else:
                cmd = f'python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions {question_str}'

            # Run subprocess using prepared command line
            returned_output = subprocess.check_output(cmd)

            # Process output
            answer = parse_cmd_output(returned_output)

            # Write output to file
            if not started_write:
                with open('answer.txt', 'w') as file:
                    file.write(f'{answer}\n\n\n')
                    started_write = True
            else:
                with open('answer.txt', 'a') as file:
                    file.write(f'{answer}\n\n\n')


if __name__ == "__main__":
    main()
