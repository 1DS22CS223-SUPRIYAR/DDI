# wrapper.py
import sys
import subprocess

# Add the dynamic patch directly to train.py via command-line injection
patch = """
import sys
if 'config' in globals():
    config['num_layers'] = config.get('n_layers')
"""

# Execute train.py with the patch prepended
subprocess.run(["python3", "-c", patch + open("train.py").read()])
