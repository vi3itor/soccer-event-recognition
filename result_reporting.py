import os
import json

# Time window (in frames) to consider the events classified correctly
#  by looking TimeWindow frames ahead and before
TimeWindow = 3


class Events:
    def __init__(self, total, passes, failed_passes, shots):
        self.total = total
        self.passes = passes
        self.failed_passes = failed_passes
        self.shots = shots

    def to_dict(self):
        return {'total': self.total, 'passes': self.passes, 'failed_passes': self.failed_passes, 'shots': self.shots}


def do_comparison(obtained_output, expected_output, include_counters=False):
    """
    Compare two outputs and return precision, recall and tp, fp, fn counters.
    :return: 
    """
    # Rounding for precision, recall and f-score
    rounding = 4

    def get_precision(true_pos, false_pos):
        return round(true_pos / (true_pos + false_pos), rounding) if true_pos + false_pos > 0 else 0

    def get_recall(true_pos, false_neg):
        return round(true_pos / (true_pos + false_neg), rounding) if true_pos + false_neg > 0 else 0

    def get_f_score(true_pos, false_pos, false_neg):
        if true_pos + false_pos + false_neg > 0:
            return round(true_pos / (true_pos + 0.5 * (false_pos + false_neg)), rounding)
        return 0

    def get_stat_dict(true_pos, false_pos, false_neg):
        return {'precision': get_precision(true_pos, false_pos), 'recall': get_recall(true_pos, false_neg),
                'f_score': get_f_score(true_pos, false_pos, false_neg),
                'true_positive': true_pos, 'false_positive': false_pos, 'false_negative': false_neg}

    assert len(obtained_output) == len(expected_output)
    result = {'overall': {}, 'passes': {}, 'failed_passes': {}, 'shots': {}}

    expected = count_events(expected_output)
    if include_counters:
        obtained = count_events(obtained_output)
        result['counters'] = {'expected': expected.to_dict(), 'obtained': obtained.to_dict()}

    # correct, incorrect, not detected
    tp, fp, fn = 0, 0, 0
    for seq in expected_output:
        comparison = compare_events(obtained_output[seq], expected_output[seq])
        tp += comparison['true_positive']
        fp += comparison['false_positive']
        fn += comparison['false_negative']

    result['overall'] = get_stat_dict(tp, fp, fn)

    # Events that are left in obtained_output dictionary are false positive (incorrectly recognized) events
    fp_events = count_events(obtained_output)
    # Events that are left in expected_output dictionary are false negative (not detected) events
    fn_events = count_events(expected_output)
    if include_counters:
        result['counters'].update({'false_positive': fp_events.to_dict(),
                                   'false_negative': fn_events.to_dict()})

    tp_passes = expected.passes - fn_events.passes
    result['passes'] = get_stat_dict(tp_passes, fp_events.passes, fn_events.passes)

    tp_failed_passes = expected.failed_passes - fn_events.failed_passes
    result['failed_passes'] = get_stat_dict(tp_failed_passes, fp_events.failed_passes, fn_events.failed_passes)

    tp_shots = expected.shots - fn_events.shots
    result['shots'] = get_stat_dict(tp_shots, fp_events.shots, fn_events.shots)
    return result


def report_results(findings):
    counters = findings.get('counters')
    if counters:
        def print_event_counters(categ, message):
            print(message)
            print(f"\tPasses: {categ['passes']}, failed passes: {categ['failed_passes']}, shots: {categ['shots']}")
            print(f"\tTotal number of events: {categ['total']}")

        print_event_counters(counters['expected'], "Expected events:")
        print_event_counters(counters['obtained'], "Obtained events:")
        print_event_counters(counters['false_positive'], "False Positive (recognized incorrectly) events:")
        print_event_counters(counters['false_negative'], "False Negative (not detected) events:")

    def print_stat(category, message=None):
        if message is not None:
            print(message)
        print(f'\t\tPrecision: {round(category["precision"] * 100, 2)}%')
        print(f'\t\tRecall: {round(category["recall"] * 100, 2)}%')
        print(f'\t\tF-Score: {round(category["f_score"] * 100, 2)}%')

    print('\nPrecision and recall by category:')
    print_stat(findings['passes'], "\tPasses:")
    print_stat(findings['failed_passes'], "\tFailed Passes:")
    print_stat(findings['shots'], "\tShots:")

    overall = findings['overall']
    print(f'\nOverall statistics:')
    print(f'Recognized correctly: {overall["true_positive"]}, incorrectly: {overall["false_positive"]}; '
          f'not detected: {overall["false_negative"]}.')
    print_stat(overall)


def compare_events(obtained, expected):
    """
    Compare two dictionaries of events, obtained by the algorithm and labelled manually.
    Remove matching events from the dictionaries.

    To calculate precision and recall we need
     - true positives: correctly classified events
     - false positives: incorrectly classified events
     - false negatives: expected events, that were not detected
    Return dictionary with number of correctly/incorrectly classified and not detected events.
    """

    result = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    # Iterate over a list of keys (frames) to be able to delete them from the dictionary
    for frame in list(expected):
        exp_event = expected[frame]
        if find_match(frame, exp_event, obtained):
            result['true_positive'] += 1
            del expected[frame]

    # TODO: If events that have a partial match - in particular the same frame number and/or same player (id) -
    #  count as false_positive (incorrect) and remove from 'expected' dictionary
    # Count events that are left in `expected` as not detected
    result['false_negative'] += len(expected)
    # Count events that are left in `obtained` as incorrect
    result['false_positive'] += len(obtained)
    return result


def find_match(frame, exp_event, obtained_events):
    """
    Find the expected event in the obtained events by looking not only
    at the frame itself, but also at the adjacent frames. As soon as match is found,
    remove the event from obtained_events dictionary and return True.
    """
    def is_event_found(fr):
        obt_event = obtained_events.get(fr)
        if obt_event is not None:
            if obt_event == exp_event:
                del obtained_events[fr]
                return True
            # print(f'Mismatch on frame {fr}!\n Obtained: {obt_event}.\n Expected: {exp_event}.')
        return False

    if is_event_found(frame):
        return True

    # Compile a list of adjacent frames to look:
    adj_frames = []
    int_frame = int(frame)
    for i in range(1, TimeWindow + 1):
        adj_frames.extend([str(int_frame + i), str(int_frame - i)])

    for adj_frame in adj_frames:
        if is_event_found(adj_frame):
            return True
    return False


def count_events(output):
    """
    Count event types in the output dictionary.
    """
    events = Events(0, 0, 0, 0)
    for seq in output:
        for frame, event in output[seq].items():
            event_type = event['event']
            if event_type == 'pass':
                events.passes += 1
            elif event_type == 'failed_pass':
                events.failed_passes += 1
            elif event_type == 'shot':
                events.shots += 1
            events.total += 1
        assert events.total == (events.passes + events.failed_passes + events.shots)
    return events
