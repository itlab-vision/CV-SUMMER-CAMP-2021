#!/usr/bin/env python3
import sys
from pathlib import Path
import platform
import subprocess
import shutil
import os

import main

if platform.system() == "Windows":
    # usually Windows terminals do not support ANSI colors
    GREENCOLOR = ''
    REDCOLOR = ''
    ENDCOLOR = ''
else:
    # usually Linux terminals support ANSI colors
    GREENCOLOR = '\033[92m'
    REDCOLOR = '\033[91m'
    ENDCOLOR = '\033[0m'

def run_test(test_config):
    dst_path = test_config["dst_path"]
    annotation_file = test_config["annotation_file"]
    probability_miss_detection = test_config["probability_miss_detection"]
    python_path = sys.executable
    if dst_path.exists():
        print("Before removing {}".format(dst_path))
        shutil.rmtree(str(dst_path))
        print("After removing {}".format(dst_path))

    cmd = [python_path,
            main.__file__,
            "--annotation", str(annotation_file),
            "--dst_folder", str(dst_path),
            #"--max_frame_index", "100", # for debugging only
            "--probability_miss_detection", str(probability_miss_detection)
            ]
    print("Running cmd =\n{}".format(" ".join(cmd)))
    subprocess.run(cmd)
    eval_file = dst_path / "evaluation.txt"
    result = None
    prefix = "recall:"
    try:
        with eval_file.open() as f_eval:
            for line in f_eval:
                if not line.startswith(prefix):
                    continue
                assert result is None, "Only one recall line should be in the evaluation file!"
                line = line[len(prefix):]
                line = line.strip()
                result = float(line)
    except:
        print("Exception is caught during handling the file {}".format(eval_file))
    return result

def test_main():
    data_folder = Path(__file__).parent / '..' / '..' / 'data'
    TEST_CONFIGS = [
            {
                "dst_path": data_folder / "DST1",
                "annotation_file": data_folder / "annotation" / "annotation.txt",
                "probability_miss_detection": 0.05,
                "target_quality": 0.9
            },
            {
                "dst_path": data_folder / "DST2",
                "annotation_file": data_folder / "annotation" / "annotation.txt",
                "probability_miss_detection": 0.6,
                "target_quality": 0.75
            },
            {
                "dst_path": data_folder / "DST3",
                "annotation_file": data_folder / "annotation" / "annotation.txt",
                "probability_miss_detection": 0.75,
                "target_quality": 0.45
            },
            ]
    num_tests = len(TEST_CONFIGS)
    test_results = []
    num_passed = 0
    total_result = True
    for i, test_config in enumerate(TEST_CONFIGS):
        test_res = {}
        test_res["config"] = test_config
        measured_quality = run_test(test_config)
        test_res["measured_quality"] = measured_quality
        target_quality = test_config["target_quality"]
        passed = (measured_quality is not None) and (measured_quality >= target_quality)
        test_res["passed"] = passed
        if passed:
            result_color = GREENCOLOR
            result_word = "PASSED"
            num_passed += 1
        else:
            result_color = REDCOLOR
            result_word = "FAILED"
            total_result = False
        print(result_color + "\n"
              + "Test {}/{} {}: {} vs {}".format(i+1, num_tests, result_word, measured_quality, target_quality)
              + "\n" + ENDCOLOR)

    print("")
    if total_result:
        result_color = GREENCOLOR
        total_result_str = "PASSED"
    else:
        result_color = REDCOLOR
        total_result_str = "FAILED"

    print(result_color + "\n"
          + "TOTAL RESULT: passed {}/{} -- {}".format(num_passed, num_tests, total_result_str)
          +"\n" + ENDCOLOR)

    if total_result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    test_main()
