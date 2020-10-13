import os
import pickle
import json
from event_recognition import EventRecognizer
from result_reporting import report_results, do_comparison


def process_file(input_file, print_output=False, show_info=False, show_debug=False):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    output = {}
    for seq in data:
        if print_output:
            print(f'Processing {seq}')
        recognizer = EventRecognizer(data[seq], show_info, show_debug)
        output[seq] = recognizer.find_events()
    if print_output:
        print(json.dumps(output, indent=2))
    return output


def save_output(input_file, output):
    """Create a file based on input file name and save output as json."""
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join('output', base + "_out.json")
    with open(output_file, 'w+') as f:
        f.write(json.dumps(output, indent=2))


def debug_algorithm(seq):
    test_file = os.path.join('input', 'test_data.pkl')
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
    print(f'Processing {seq}')
    recognizer = EventRecognizer(data[seq], show_info=True, show_debug=True)
    obtained_output = {seq: recognizer.find_events()}
    print(json.dumps(obtained_output, indent=2))

    expected_json = os.path.join('output_expected', 'test_data_out_expected.json')
    with open(expected_json, 'r') as f:
        full = json.load(f)
        expected_output = {seq: full[seq]}
    result = do_comparison(obtained_output, expected_output, include_counters=True)
    report_results(result)


def main():
    data_file = os.path.join('input', 'test_data.pkl')
    obtained_output = process_file(data_file, print_output=False, show_info=False, show_debug=False)
    save_output(data_file, obtained_output)
    expected_json = os.path.join('output_expected', 'test_data_out_expected.json')
    with open(expected_json, 'r') as f:
        expected_output = json.load(f)
    comparison = do_comparison(obtained_output, expected_output, include_counters=True)
    report_results(comparison)


if __name__ == '__main__':
    main()
    # sequence = 'sequence_2'
    # debug_algorithm(sequence)
