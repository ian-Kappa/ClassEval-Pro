import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.runner import main
from inference import HierarchicalCodeGenPipeline as Pipeline

if __name__ == "__main__":
    main(Pipeline, default_max_length=20000, default_auto_test=1)
