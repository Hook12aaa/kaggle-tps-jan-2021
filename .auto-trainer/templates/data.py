import sys
sys.path.insert(0, "/Users/hook/Documents/coding/python/kaggle/tps-jan-2021/.auto-trainer")
import harness
from config import CONFIG


def get_split():
    return harness.load_split(CONFIG)
