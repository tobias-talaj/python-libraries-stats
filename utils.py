import os
import ast
import pickle
import logging
import warnings
import nbformat
from nbconvert import PythonExporter
from typing import List, Tuple, Dict


def setup_logger(logger_name, level=logging.ERROR):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(f'{logger_name}.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger


def find_python_files(
        root_directory: str,
        filetype: str = ".py",
        dir_range: Tuple[int, int] = (0, float("inf"))
) -> List[str]:
    """
    Traverse directories within the given root directory and return a list of paths for
    files that match the given filetype, within a range of top-level directories specified.

    Parameters:
    root_directory (str): The root directory to start the search from.
    filetype (str): The file extension to look for. Defaults to '.py'.
    dir_range (tuple): A tuple of two numbers specifying the first and last top-level
                       directory to include in the search. Defaults to (0, infinity).

    Returns:
    List[str]: A list of paths for all found files that match the given filetype.
    """
    python_files = []
    dir_counter = -1

    for dir_name in sorted(os.listdir(root_directory)):
        full_dir_name = os.path.join(root_directory, dir_name)
        if os.path.isdir(full_dir_name):
            dir_counter += 1
            if dir_counter < dir_range[0]:
                continue
            if dir_counter > dir_range[1]:
                break
            for dirpath, _, filenames in os.walk(full_dir_name):
                for filename in filenames:
                    if filename.endswith(filetype):
                        full_path = os.path.join(dirpath, filename)
                        python_files.append(full_path)

    return python_files


def convert_notebook_to_python(
        notebook_json: str,
        logger: logging.Logger
) -> str:
    """
    Convert a Jupyter Notebook (.ipynb) JSON string to a Python script, 
    excluding cells with syntax errors and lines starting with '%%'.

    Parameters:
    notebook_json: The Jupyter Notebook JSON string to be converted.
    logger: Logger object for logging messages.

    Returns:
    The Python script converted from the Jupyter Notebook JSON string.
    """
    python_script = ''

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            notebook_node = nbformat.reads(notebook_json, as_version=4)
            valid_cells = []
            for cell in notebook_node.cells:
                if cell.cell_type == 'code':
                    filtered_source = '\n'.join(line for line in cell.source.splitlines() 
                                                if not (line.strip().startswith('%') or line.strip().startswith('!')))
                    try:
                        cell.source = filtered_source
                        ast.parse(filtered_source)
                        valid_cells.append(cell)
                    except SyntaxError:
                        logger.warning(f"Skipping cell with syntax error: {cell.source}")
                    except Exception as e:
                        logger.error(f"Cannot format notebook cell: {e}")
                else:
                    valid_cells.append(cell)
            valid_notebook = nbformat.v4.new_notebook(cells=valid_cells)
            exporter = PythonExporter()
            python_script, _ = exporter.from_notebook_node(valid_notebook)
    except Exception as e:
        logger.error(f"Couldn't convert notebook to python: {e}")

    return python_script


def load_library_reference(library_pickle_path: str) -> Dict:
    with open(library_pickle_path, "rb") as f:
        lib_dict = pickle.load(f)
    return lib_dict
