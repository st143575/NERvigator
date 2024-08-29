import argparse
from pathlib import Path

def parse_arguments():
    """
    Parse command-line arguments for input and output files, and ensure the parent directory of the output file exists.

    Command-line arguments:
        - `-i` or `--input_file` (str): Path to the input file. 
          Default is '../datasets/input.txt'.
        - `-o` or `--output_file` (str): Path to the output file. 
          Default is '../cache/output.txt'.

    Returns:
        tuple: A tuple containing:
            - input_file (Path): Path object representing the input file.
            - output_file (Path): Path object representing the output file. 
              If the parent directory of the output file does not exist, it is created.

    Example:
        input_file, output_file = parse_arguments()
        # Now you can use input_file and output_file as Path objects
    """
    parser = argparse.ArgumentParser(description='Parse command-line arguments for input and output files')
    parser.add_argument('-i', '--input_file', type=str, default='../datasets/input.txt', help="Path to the input file")
    parser.add_argument('-o', '--output_file', type=str, default='../cache/output.txt', help="Path to the output file")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    # Creates output folder structure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    return input_file, output_file