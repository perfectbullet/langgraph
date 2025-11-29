import getpass
import os


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")

os.environ['DEEPSEEK_API_KEY'] = 'sk-0511c57af3604877b63cf32ea9ae7f01'

