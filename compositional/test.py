import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluate import *  # noqa: F401,F403

if __name__ == "__main__":
    from common.evaluate import _cli
    _cli()
