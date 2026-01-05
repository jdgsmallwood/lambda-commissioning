import os
import toml
import shutil
import importlib.resources as resources

HOMEDIR = os.path.expanduser("~")
configFile = "default_config.toml"
configPath = "lambda_commissioning.config"

def ensure_output_dirs():
    # Load the default config file from package resources
    with resources.files(configPath).joinpath(configFile).open("r") as f:
        config = toml.load(f)

    directoryDict = config.get("paths", {})

    for _,path in directoryDict.items():
        if not os.path.exists(HOMEDIR+path):
            os.makedirs(HOMEDIR+path,exist_ok=True)
            print(f"Created directory: {HOMEDIR+path}")