import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(HERE.parent.as_posix())  # add parent to path for importing compile_tl_op

from compile_tl_op import flash_attention

flash_attention.update_so()

