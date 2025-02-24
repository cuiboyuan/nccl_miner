'''
Parse relevant information from torch profiler data.
'''
import json
import argparse
import os

def parse_torch_logs(log_files):
    # torch_ops_per_device = {}
    # all_events = []
    # for log in log_files:
    #     print(f"Parsing log file {log}..")
    #     cpu_ops = []
    #     kernel_ops = []
    #     with open(log, "r") as f:
    #         trace = json.load(f)
    #         events = trace['traceEvents']
    #         all_events += events
    # out_json = {'traceEvents': all_events}
    # with open("torch_combined.json", "w") as f:
    #     json.dump(out_json, f)
    # print(f"Written to file")
    return {}, {}

# Local testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="The directory which contains torch logs to parse.")
    parser.add_argument("-o", "--output_file", default="torch_trace.json", type=str, help="The name of the output trace JSON file.")
    args = parser.parse_args()

    torch_log_files = []
    for log_file in os.listdir(args.log_dir):
        log_path = os.path.join(args.log_dir, log_file)
        if os.path.isfile(log_path):
            torch_log_files.append(log_path)
    print(torch_log_files)
    parse_torch_logs(torch_log_files)