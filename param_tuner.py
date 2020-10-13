import os
import json
import pickle
import csv
import math
import random
import numpy as np
from copy import deepcopy

import event_recognition
from result_reporting import do_comparison


def load_files():
    data_file = os.path.join('input', 'test_data.pkl')
    labels_file = os.path.join('output_expected', 'test_data_out_expected.json')
    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    # Load labels
    with open(labels_file, 'r') as f:
        expected_output = json.load(f)
    return data, expected_output


def run_and_compare(data, labels):
    obtained_output = {}
    for seq in data:
        recognizer = event_recognition.EventRecognizer(data[seq], show_info=False)
        obtained_output[seq] = recognizer.find_events()
    expected_output = deepcopy(labels)
    comparison = do_comparison(obtained_output, expected_output)
    return comparison


def tune_params_independently(full_info=False):
    params = {
        'GracePeriodPlayer': list(range(1, 21)),
        'VicinityThreshold': np.linspace(0.1, 3.0, 30),
        'MinFailedPassLength': np.linspace(0.5, 7.5, 71),
        'MinTrChangeAngle': np.linspace(0.1, math.pi, 36),  # 5 degrees difference
        'MinSpeedChangeFactor': np.linspace(1.1, 2.1, 11)
    }

    data, labels = load_files()

    for param_name, values in params.items():
        rows = []
        default_value = getattr(event_recognition, param_name)
        for value in values:
            value = round(value, 2)
            setattr(event_recognition, param_name, value)
            print(f'Setting {param_name} to {value}...')

            comparison = run_and_compare(data, labels)
            row = {f'{param_name}': value,
                   'precision': comparison['overall']['precision'],
                   'recall': comparison['overall']['recall'],
                   'f-score': comparison['overall']['f_score']}
            if full_info:
                row.update({'passes_precision': comparison['passes']['precision'],
                            'passes_recall': comparison['passes']['recall'],
                            'passes_f_score': comparison['passes']['f_score'],
                            'failed_passes_precision': comparison['failed_passes']['precision'],
                            'failed_passes_recall': comparison['failed_passes']['recall'],
                            'failed_passes_f_score': comparison['failed_passes']['f_score'],
                            'shots_precision': comparison['shots']['precision'],
                            'shots_recall': comparison['shots']['recall'],
                            'shots_f_score': comparison['shots']['f_score']})
            rows.append(row)

        setattr(event_recognition, param_name, default_value)
        # Save as csv
        filename = os.path.join('output', 'csv', f'{"full_" if full_info else ""}{param_name}.csv')
        csv_columns = rows[0].keys()
        with open(filename, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == '__main__':
    tune_params_independently(full_info=True)
