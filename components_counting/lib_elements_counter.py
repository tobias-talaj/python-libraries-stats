import ast
import logging
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Callable

import pandas as pd

from components_counting.utils import convert_notebook_to_python


def get_imported_libs(
    tree: ast.AST,
    consolidate_imports: bool = True,
    max_depth: int = 2
) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    """
    Traverses the AST up to a maximum depth and identifies all import statements,
    returning a set of imported library names and directly imported components.

    Parameters:
    tree: The root of the Abstract Syntax Tree (AST).
    consolidate_imports: Consolidates all module imports into one library import e.g. sklearn.linear_model and sklearn become just sklearn.
    max_depth: The maximum depth to traverse in the AST.

    Returns:
    A tuple containing:
    - A set of strings, each string being the name of an imported library.
    - A dictionary mapping directly imported component names to their library names.
    - A dictionary mapping modules and submodules aliases to their library names.
        (e.g. import sklearn.linear_model as lm, import sklearn as sk -> {'sklearn': {'lm', 'sk'}})
    """
    imported_libs = set()
    direct_imports = defaultdict(set)
    aliases = defaultdict(set)

    def walk_tree(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return

        match node:
            case ast.Import(names=names):
                lib_names = [n.name.split('.')[0] if consolidate_imports else n.name for n in names]
                imported_libs.update(lib_names)
                for n in names:
                    aliases[n.name.split('.')[0]].add(n.asname)
            case ast.ImportFrom(module=lib, names=names, level=0):
                lib_name = lib.split('.')[0] if consolidate_imports else lib
                imported_libs.add(lib_name)
                direct_imports[lib_name].update(n.name for n in names)
                aliases[lib_name].update(n.asname for n in names if n.asname is not None)

        for child in ast.iter_child_nodes(node):
            walk_tree(child, depth + 1)

    walk_tree(tree)
    return imported_libs, direct_imports, aliases


def get_expression_elements(node: ast.AST) -> List[str]:
    """
    Returns a list of object names used in given chain.
    e.g. foo(a=b).bar(c=d).baz() -> ['foo', 'bar', 'baz']
    """
    unparsed = ast.unparse(node)
    stack = []
    result = []
    for char in unparsed:
        if char == '(':
            stack.append(len(result))
        elif char == ')' and stack:
            start = stack.pop()
            result = result[:start]
        elif not stack:
            result.append(char)
    return ''.join(result).split('.')


def get_object_names(tree: ast.AST, classes: Set[str]) -> Set[str]:
    """
    Traverses the AST and identifies object names of given classes.
    Goes two levels deep, that means b = A(); c = b.foo() will give {b, c}.

    Parameters:
    tree: The root of the Abstract Syntax Tree (AST).
    classes: Collection of strings with class names.

    Returns:
    A set of strings, each string being the name of a class instantiation.
    """
    object_names = set()
    def walk_tree(node, classes):
        match node:
            case ast.Assign(targets=targets, value=object_instantiation)\
                if isinstance(targets[0], ast.Tuple)\
                and any(c in get_expression_elements(object_instantiation) for c in classes):
                object_names.update([e.id for e in node.targets[0].elts])
            case ast.Assign(value=object_instantiation)\
                if isinstance(targets[0], ast.Name)\
                and any(c in get_expression_elements(object_instantiation) for c in classes):
                object_names.add(node.targets[0].id)
        for child in ast.iter_child_nodes(node):
            walk_tree(child, classes)
    walk_tree(tree, classes)
    walk_tree(tree, object_names)
    return object_names


def check_node(
    node: ast.AST, 
    components: Dict[str, List[str]], 
    df: pd.DataFrame, 
    code_file: str, 
    library: str, 
    library_direct_imports: Set[str], 
    object_names: List[str], 
    library_aliases: Set[str]
) -> None:
    """
    Checks if the node represents any of the library components (functions, methods, classes instatiations, attributes, and exceptions).
    If it does, it updates the given DataFrame with the component's count.

    Parameters:
    node: The AST node to check.
    components: A dict containing API reference of a given library.
    df: A DataFrame object for storing counts of library components.
    code_file: The path to the Python file being processed.
    library: The name of the library which components are being checked.
    library_direct_imports: A set of directly imported component names.
    library_aliases: A set of aliases for given library and its modules.


    Returns:
    None
    """
    match node:
        case ast.Call(func=ast.Name(id=func_name))\
            if func_name in components["function"]\
            and func_name in library_direct_imports:
                update_df(df, code_file, library, "function", func_name)
        case ast.Call(func=ast.Attribute(attr=func_name, value=ast.Name(id=library_name)))\
            if func_name in components["function"]\
            and (library_name == library or library_name in components["module"] or library_name in library_aliases):
                update_df(df, code_file, library, "function", func_name)

        case ast.Call(func=ast.Attribute(attr=method_name))\
            if method_name in components["method"]\
            and any(o in get_expression_elements(node) for o in object_names):
                update_df(df, code_file, library, "method", method_name)
        case ast.Call(func=ast.Attribute(attr=method_name))\
            if method_name in components["method"]\
            and any(c in get_expression_elements(node) for c in components["class"]):
                update_df(df, code_file, library, "method", method_name)
        case ast.Call(func=ast.Attribute(attr=method_name))\
            if method_name in components["method"]\
            and any(a in get_expression_elements(node) for a in library_aliases):
                update_df(df, code_file, library, "method", method_name)
        case ast.Call(func=ast.Attribute(attr=method_name))\
            if method_name in components["method"]\
            and library == get_expression_elements(node)[0]:
                update_df(df, code_file, library, "method", method_name)

        case ast.Call(func=ast.Name(id=class_name))\
            if class_name in components["class"]\
            and class_name in library_direct_imports:
                update_df(df, code_file, library, "class", class_name)
        case ast.Call(func=ast.Attribute(value=ast.Name(id=library_name), attr=class_name))\
            if class_name in components["class"]\
            and (library_name == library or library_name in components["module"] or library_name in library_aliases):
                update_df(df, code_file, library, "class", class_name)

        case ast.Attribute(attr=attr_name)\
            if attr_name in components["attribute"]\
            and any(o in get_expression_elements(node) for o in object_names):
                update_df(df, code_file, library, "attribute", attr_name)
        case ast.Attribute(attr=attr_name)\
            if attr_name in components["attribute"]\
            and any(c in get_expression_elements(node) for c in components["class"]):
                update_df(df, code_file, library, "attribute", attr_name)
        case ast.Attribute(attr=attr_name)\
            if attr_name in components["attribute"]\
            and any(a in get_expression_elements(node) for a in library_aliases):
                update_df(df, code_file, library, "attribute", attr_name)
        case ast.Attribute(attr=attr_name)\
            if attr_name in components["attribute"]\
            and library == get_expression_elements(node)[0]:
                update_df(df, code_file, library, "attribute", attr_name)
            
        case ast.ExceptHandler(type=ast.Name(id=exc_name))\
            if exc_name in components["exception"]\
            and exc_name in library_direct_imports:
                update_df(df, code_file, library, "exception", exc_name)
        case ast.ExceptHandler(type=ast.Attribute(value=ast.Name(id=library_name), attr=exc_name))\
            if exc_name in components["exception"]\
            and library_name == library:
                update_df(df, code_file, library, "exception", exc_name)
        case ast.Raise(exc=ast.Call(func=ast.Name(id=exc_name)))\
            if exc_name in components["exception"]\
            and exc_name in library_direct_imports:
                update_df(df, code_file, library, "exception", exc_name)
        case ast.Raise(exc=ast.Call(func=ast.Attribute(value=ast.Name(id=library_name), attr=exc_name)))\
            if exc_name in components["exception"]\
            and library_name == library:
                update_df(df, code_file, library, "exception", exc_name)

        case ast.Call(args=args, keywords=keywords):
            for arg in args:
                if isinstance(arg, ast.Name) and arg.id in components["function"] and arg.id in library_direct_imports:
                    func_name = arg.id
                    update_df(df, code_file, library, "function", func_name)
                elif isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == library:
                    func_name = arg.attr
                    if func_name in components["function"]:
                        update_df(df, code_file, library, "function", func_name)
            for kw in keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id in components["function"] and kw.value.id in library_direct_imports:
                    func_name = kw.value.id
                    update_df(df, code_file, library, "function", func_name)
                elif isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name) and kw.value.value.id == library:
                    func_name = kw.value.attr
                    if func_name in components["function"]:
                        update_df(df, code_file, library, "function", func_name)


def update_df(df: pd.DataFrame, code_file: str, library: str, component_type: str, component_name: str) -> None:
    """
    Checks whether a row for given code file, library, component type and component name already exists.
    If it does, the count in that row is incremented. 
    If it doesn't, a new row is added to the DataFrame with a count of 1.

    Parameters:
    df: The DataFrame to be updated. 
        It has the following columns:
        - 'filename': the path to the Python file
        - 'library': the name of the library
        - 'component_type': the type of the component (e.g., 'function', 'class', etc.)
        - 'component_name': the name of the component
        - 'count': the count of the component
    code_file: The path to the Python file being processed.
    library: The name of the library which components are being checked.
    component_type: The type of the component (e.g., 'function', 'class', etc.).
    component_name: The name of the component.

    Returns:
    None
    """
    row_exists = ((df['filename'] == code_file) & 
                  (df['library'] == library) & 
                  (df['component_type'] == component_type) & 
                  (df['component_name'] == component_name)).any()
    if row_exists:
        df.loc[(df['filename'] == code_file) & 
               (df['library'] == library) & 
               (df['component_type'] == component_type) & 
               (df['component_name'] == component_name), 'count'] += 1
    else:
        new_row = {'filename': code_file,
                   'library': library,
                   'component_type': component_type,
                   'component_name': component_name,
                   'count': 1}
        df.loc[len(df)] = new_row


def process_file_full_analysis(logger: logging.Logger, lib_dict: Dict, code_file: str) -> pd.DataFrame:
    """
    Process a single file, returning a DataFrame with counts of library components.

    Parameters:
    logger: Logger object for logging messages.
    lib_dict: A dictionary representing the API reference of one or more libraries.
    code_file: The path to the file to process.

    Returns:
    A DataFrame containing counts of library components.
    """
    columns = ['filename', 'library', 'component_type', 'component_name', 'count']
    df = pd.DataFrame(columns=columns)
    try:
        with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except IOError as e:
        logger.error(f"Error reading file {code_file}: {e}")
        return pd.DataFrame(columns=columns)
    
    if code_file.endswith('.ipynb'):
        logger.debug(f"Trying to convert {code_file} to regular Python file.")
        code = convert_notebook_to_python(code, logger)

    try:
        logger.debug(f"Counting components {code_file}.")
        tree = ast.parse(code)
        imported_libraries, direct_imports, aliases = get_imported_libs(tree)
        
        for library, components in lib_dict.items():
            if library not in imported_libraries:
                continue
            object_names = get_object_names(tree, components['class'])
            for node in ast.walk(tree):
                check_node(node, components, df, code_file, library, direct_imports[library], object_names, aliases[library])
        logger.debug(f"Successfully counted components in {code_file}. Dataframe: \n{df}")
        return df
    except SyntaxError as e:
        logger.error(f"Syntax error parsing file {code_file}: {e}")
        return pd.DataFrame(columns=columns)
    except Exception as e:
        logger.error(f"Exception {code_file}: {e}")
        return pd.DataFrame(columns=columns)
    

def process_file_simple_analysis(logger: logging.Logger, code_file: str, _) -> pd.DataFrame:
    """
    Process a single file, returning a DataFrame with imported libraries.

    Parameters:
    logger: Logger object for logging messages.
    code_file: The path to the file to process.

    Returns:
    A DataFrame with filenames and imported libraries within the given code file.
    """
    columns = ['filename', 'library']
    df = pd.DataFrame(columns=columns)
    try:
        with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except IOError as e:
        logger.error(f"Error reading file {code_file}: {e}")
        return pd.DataFrame(columns=columns)

    if code_file.endswith('.ipynb'):
        code = convert_notebook_to_python(code, logger)

    try:
        tree = ast.parse(code)
        imported_libraries, _ = get_imported_libs(tree)

        for library in imported_libraries:
            new_row = {'filename': code_file, 'library': library}
            df.loc[len(df)] = new_row
        return df
    except SyntaxError as e:
        logger.error(f"Syntax error parsing file {code_file}: {e}")
        return pd.DataFrame(columns=columns)
    except Exception as e:
        logger.error(f"Exception {code_file}: {e}")
        return pd.DataFrame(columns=columns)


def process_files_in_parallel(
    process_file_func: Callable[[logging.Logger, Dict, str, str], pd.DataFrame],
    lib_dict: Dict,
    code_files: List[str],
    logger: logging.Logger
) -> List[pd.DataFrame]:
    """
    Process given files in parallel, returning a list of DataFrames.

    Parameters:
    process_file_func: Function to be applied to each file.
    lib_dict: A dictionary representing the library.
    code_files: A list of paths to Python code files.
    logger: Logger object for logging messages.
    mode: Mode of operation, 'full' for full analysis or 'imports' for filenames and imports only.

    Returns:
    A list of DataFrames, each resulting from processing a single file.
    """
    process_file_partial = partial(process_file_func, logger, lib_dict)
    with Pool() as pool:
        results = pool.map(process_file_partial, code_files)
    print(f'Number of DataFrames: {len(results)}')
    return [df for df in results if not df.empty]


def concatenate_and_save(df_list: List[pd.DataFrame], output_file: str) -> None:
    """
    Concatenate given list of DataFrames and save the results to a parquet file.

    If the DataFrames contain a 'count' column, the function will group by the 'filename', 'library', 
    'component_type', and 'component_name' columns and sum the 'count' column. If the 'count' column 
    does not exist, the function will simply concatenate the DataFrames.

    Parameters:
    df_list: A list of DataFrames.
    output_file: Path to the output parquet file.

    Returns:
    None
    """
    df_concat = pd.concat(df_list)
    if 'count' in df_concat.columns:
        df_final = df_concat.groupby(['filename', 'library', 'component_type', 'component_name'], as_index=False).sum()
    else:
        df_final = df_concat
    df_final.to_parquet(output_file, engine="pyarrow")
