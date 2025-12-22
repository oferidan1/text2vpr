"""
Compatibility wrapper.

The main implementation lives in `sam3_checkr.py` (historical filename).
This module exists so users can run:

    python3 sam3_checker.py --input_csv ...
"""

from sam3_checkr import main


if __name__ == "__main__":
    main()





