import argparse
import subprocess
import re


ANS_PATTERN = re.compile(
    r'---answer: (?P<score>\d.\d\d) (?P<answer>[\S ]+)\s*.*   (?P<context>[\S ]+)',
    re.MULTILINE,
)


def parse_cmd_output(text):
    # Convert bytes to str
    text: str = text.decode('cp437')

    # Search for all answers
    questions = []
    for line in text.splitlines(keepends=False):
        match = re.match(r'^\[ INFO \] Question: (.+)$', line)
        if match:
            questions.append((match.group(1), []))
            continue
        match = re.match(r'^\[ INFO \] ---answer: (\d\.\d\d) (.+)$', line)
        if match:
            questions[-1][1].append(match.group(2))
            continue

    ret = []
    for question, answers in questions:
        ret.append(f'{question} --- {" / ".join(answers)}')
    ret = '\n'.join(ret)

    return ret


def build_argparser():
    parser = argparse.ArgumentParser(description="Simple object tracker demo")
    parser.add_argument("-q", required=True, help="Path to the questions file")
    parser.add_argument("-i", required=True, help="Path to the input sites file")
    parser.add_argument("-m", required=True, help="Path to the model")
    return parser


def main():
    args = build_argparser().parse_args()

    # Load input data
    with open(args.q) as fp:
        questions = [line.strip() for line in fp]
    with open(args.i) as fp:
        sites = [line.strip() for line in fp]

    answer = []
    for site in sites:
        # Prepare input parameters for script
        path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
        path_to_model = args.m
        questions_str = ' '.join(f'"{question}"' for question in questions)

        # Prepare text command line
        cmd = f'./venv/Scripts/python "{path_to_demo}" -v vocab.txt -m {path_to_model} --input="{site}" --questions {questions_str}'

        # Run subprocess using prepared command line
        returned_output = subprocess.check_output(cmd)

        # Process output
        answer.append(f'[{site}]\n{parse_cmd_output(returned_output)}\n')

    # Write output to file
    answer = '\n'.join(answer)
    with open('answer.txt', 'w') as the_file:
        the_file.write(f'{answer}')


if __name__ == "__main__":
    main()
