import os
import ast
import pickle
import logging
import warnings
import nbformat
from typing import List, Dict
from nbconvert import PythonExporter
from nbformat.notebooknode import NotebookNode


def setup_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(f'{logger_name}.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger


def find_python_files(root_directory: str, filetype: str = ".py") -> List[str]:
    """
    Traverse directories within the given root directory and return a list of paths for files that match the given filetype.

    Parameters:
    root_directory (str): The root directory to start the search from.
    filetype (str): The file extension to look for. Defaults to '.py'.

    Returns:
    List[str]: A list of paths for all found files that match the given filetype.
    """
    python_files = []

    for dir_name in os.listdir(root_directory):
        full_dir_name = os.path.join(root_directory, dir_name)
        for dirpath, _, filenames in os.walk(full_dir_name):
            for filename in filenames:
                if not filename.endswith(filetype):
                    continue
                full_path = os.path.join(dirpath, filename)
                python_files.append(full_path)

    return python_files


def _filter_valid_cells(notebook_node, logger: logging.Logger) -> NotebookNode:
    """
    Filters out code cells with syntax errors and lines starting with '%%' or '!'.

    Parameters:
    notebook_node: The Jupyter Notebook node to be filtered.
    logger: Logger object for logging messages.

    Returns:
    A new notebook node with only valid cells.
    """
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
    return nbformat.v4.new_notebook(cells=valid_cells)


def convert_notebook_to_python(notebook_json: str, logger: logging.Logger) -> str:
    """
    Convert a Jupyter Notebook (.ipynb) JSON string to a Python script, 
    excluding cells with syntax errors and lines starting with '%%'.

    Parameters:
    notebook_json: The Jupyter Notebook JSON string to be converted.
    logger: Logger object for logging messages.

    Returns:
    The Python script converted from the Jupyter Notebook JSON string.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            notebook_node = nbformat.reads(notebook_json, as_version=4)
            valid_notebook = _filter_valid_cells(notebook_node, logger)
            exporter = PythonExporter()
            python_script, _ = exporter.from_notebook_node(valid_notebook)
    except Exception as e:
        python_script = ''
        logger.error(f"Couldn't convert notebook to python: {e}")

    return python_script


def load_library_reference(library_pickle_path: str) -> Dict:
    with open(library_pickle_path, "rb") as f:
        lib_dict = pickle.load(f)
    return lib_dict
