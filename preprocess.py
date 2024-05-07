import os
import struct
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import pandas as pd


def load_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".dat")]
    data = []

    for file_name in files:
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'rb') as file:
            # Read the binary data and unpack it according to your specifications
            file_content = file.read()
            samples = struct.unpack('<' + 'h' * (len(file_content) // 2), file_content)
            if file_name[0] == 'n':
                label = 2
            elif file_name[0] == 's':
                label = 1
            else:
                label = 0

            # Add the data to your list or process it as needed
            data.append({
                'file_name': file_name,
                'samples': samples,
                'label': label
            })

    return data

def filter_signal(signals, sampling_rate, print_flag = False):
    results = []
    for sig in signals:
        ecg_result = ecg.ecg(signal=sig, sampling_rate=sampling_rate, show=False)
        results.append(ecg_result['filtered'])
        if print_flag:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

            # Plot original signal
            axes[0].plot(ecg_result['ts'], sig, label='Original ECG', color='orange')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            axes[0].set_title('Original ECG Signal')
            axes[0].grid(True)

            # Plot filtered signal
            axes[1].plot(ecg_result['ts'], ecg_result['filtered'], label='Filtered ECG', color='blue')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Amplitude')
            axes[1].set_title('Filtered ECG Signal')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

            # Plot QRS complexes
            plt.figure(figsize=(10, 4))
            plt.plot(ecg_result['ts'], ecg_result['filtered'], label='ECG Signal')
            plt.scatter(ecg_result['ts'][ecg_result['rpeaks']], ecg_result['filtered'][ecg_result['rpeaks']], color='red',
                        label='QRS Complexes')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('ECG Signal with Detected QRS Complexes')
            plt.legend()
            plt.grid(True)
            plt.show()
    return results


def segment_and_label_signal(row, sig_label, segment_t, overlap_t, fs):
    segment_length = segment_t * fs
    overlap = overlap_t * fs

    bias_s = row['label'] * 60  # Use bias from the specified field in the row

    segments = []
    labels = []
    start_index = 0
    while start_index + segment_length <= len(row[sig_label]):
        segment = row[sig_label][start_index:start_index + segment_length]
        segments.append(segment)
        label = bias_s + (len(row[sig_label])/fs - start_index/fs)
        labels.append(label)
        start_index += segment_length - overlap

    segmented_df = pd.DataFrame(
        {'Segment': segments,
         'Label': labels,
         'True_label': row['label'],
         'patient': row.name})

    return segmented_df

def load_preprocess(dir, fs, segment_time):
    loaded_data = load_files(dir)
    df = pd.DataFrame(loaded_data)
    # filtered for example:
    filter_signal(df.samples.head(1).apply(lambda arr: arr[:fs * 7]), fs, False)

    # filtering:
    df['filtered_sig'] = filter_signal(df.samples, fs, False)

    # create short signal with new label for regression
    segment_t = segment_time # [sec]
    overlap_t = 0  # [sec]
    s_size = segment_t * fs

    seg_signal = df.apply(
        lambda row: segment_and_label_signal(row, 'filtered_sig', segment_t, overlap_t, fs),
        axis=1)
    seg_data = pd.concat(seg_signal.tolist(), ignore_index=True)
    return seg_data, s_size