import subprocess


def parse_cmd_output(input):
    # Cut technical information from cmd output
    text = str(input)
    text = text.split("[ INFO ]")

    res = ""
    for i in range(len(text)):
        text[i] = text[i].strip()
        if text[i][:10] == "Question: ":
            res += ("\n" + text[i] + "\n")
        elif text[i][:11] == "---answer: ":
            res += (text[i] + "\n")
    print(res)
    return res


def main():

    # Prepare input parameters for script 
    path_to_demo = "C:/Program Files (x86)/Intel/openvino_2021/deployment_tools/open_model_zoo/demos/bert_question_answering_demo/python/bert_question_answering_demo.py"
    path_to_model = "E:/Downloads/intel/CV-SUMMER-CAMP-2021/practice/intel/bert-small-uncased-whole-word-masking-squad-0001/FP16/bert-small-uncased-whole-word-masking-squad-0001.xml"
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
