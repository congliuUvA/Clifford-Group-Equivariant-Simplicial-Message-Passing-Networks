import itertools
import subprocess
import sys

import yaml


def git_detached():
    # Get the output of 'git status'
    git_status_output = subprocess.getoutput("git status")
    return "HEAD detached" in git_status_output


def main():
    if git_detached():
        input("WARNING: You are in a detached HEAD state. Press enter to continue.")
    argv = sys.argv
    config = argv[1]
    args = argv[2:]

    with open(config) as f:
        if not config.endswith(".yaml") and not config.endswith(".yml"):
            raise ValueError("Config file must be a YAML file.")
        config = yaml.load(f, yaml.SafeLoader)

    parameters = config["parameters"]
    base_command = config["command"]
    for i, c in enumerate(base_command):
        if c == "${env}":
            base_command[i] = "/usr/bin/env"
        elif c == "${interpreter}":
            base_command[i] = "python -u"
        elif c == "${program}":
            base_command[i] = config["program"]
        elif c == "${args}":
            del base_command[i]

    for k, v in parameters.items():
        parameters[k] = parameters[k]["values"]

    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for d in permutations_dicts:
        print("\nRunning with configuration:")
        print(yaml.dump(d))
        print()
        command = base_command + [f"--{k}={v}" for k, v in d.items()]
        command = " ".join(command + args)
        result = subprocess.call(command, shell=True)

        if result != 0:
            break


if __name__ == "__main__":
    main()
