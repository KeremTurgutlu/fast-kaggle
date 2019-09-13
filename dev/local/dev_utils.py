from pprint import pprint
import subprocess
import shlex

__all__ = ["parse_ml_params", "process_eval_input", "run_command"]

def parse_ml_params(ml_task_params): 
    ml_params = {l['key']: l['value'] for l in ml_task_params}
    # None - null value is not supported in experiments
    for k,v in ml_params.items(): 
        if v == "None": ml_params[k] = None
    return ml_params


def process_eval_input(input_dataset, name):
    "Returns path of input dataset that corresponds to name, e.g INPUT_TRAIN_DATA_1"
    path = []
    for in_data in input_dataset:
        if in_data['name'] == name:
            path = in_data['path']
            print('Input data path: %s' % (path))
    if path == []:
        print('Required input paths not provided')
    return path

def run_command(command):
    "Run shell command as an external process"
    if type(command) == str: command = shlex.split(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print (output.decode().strip())
    rc = process.poll()
    if rc != 0:
        stdout, stderr =  process.communicate()
        return stderr.decode()
    else:
        stdout, stderr =  process.communicate()
        print(stderr.decode())
        return "Success"