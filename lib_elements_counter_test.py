import ast
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from lib_elements_counter import check_node


code = """
from xyz import direct_import_func_call, direct_import_function_as_an_argument, direct_import_second_function_as_an_argument, direct_import_function_as_a_kwarg, direct_import_second_function_as_a_kwarg, ExampleClass, ExampleException


direct_import_func_call()
xyz.func_call()

example_object.example_method()

ExampleClass()
xyz.AnotherClass()

xyz.some_object.example_attribute

try:
    some_function()
except ExampleException:
    pass
try:
    some_function()
except xyz.AnotherException:
    pass
raise ExampleException()
raise xyz.AnotherException()

example_function(direct_import_function_as_an_argument, direct_import_second_function_as_an_argument)
example_function(xyz.first_function_as_an_argument, xyz.second_function_as_an_argument)
example_function(kwarg_a=direct_import_function_as_a_kwarg, kwarg_b=direct_import_second_function_as_a_kwarg)
example_function(kwarg_a=xyz.first_function_as_a_kwarg, kwarg_b=xyz.second_function_as_a_kwarg)
"""

test_lib_api = {
    'xyz': {
        'function': ['func_call', 'direct_import_func_call', 'direct_import_function_as_an_argument', 'direct_import_second_function_as_an_argument', 'direct_import_function_as_a_kwarg', 'direct_import_second_function_as_a_kwarg', 'first_function_as_an_argument', 'second_function_as_an_argument', 'first_function_as_a_kwarg', 'second_function_as_a_kwarg'],
        'method': ['example_method'],
        'class': ['ExampleClass', 'AnotherClass'],
        'attribute': ['example_attribute'],
        'exception': ['ExampleException', 'AnotherException']
    }
}

direct_imports= {
    'direct_import_func_call': 'xyz',
    'direct_import_function_as_an_argument': 'xyz',
    'direct_import_second_function_as_an_argument': 'xyz',
    'direct_import_function_as_a_kwarg': 'xyz',
    'direct_import_second_function_as_a_kwarg': 'xyz',
    'ExampleClass': 'xyz',
    'ExampleException': 'xyz'
}

@pytest.fixture
def expected_dataframe():
    data = {
        'filename': ['test'] * 16,
        'module': ['xyz'] * 16,
        'component_type': ['function', 'function', 'method', 'class', 'class', 'attribute', 'exception', 'exception', 'function', 'function', 'function', 'function', 'function', 'function', 'function', 'function'],
        'component_name': [
            'direct_import_func_call', 'func_call', 'example_method', 'ExampleClass', 'AnotherClass',
            'example_attribute', 'ExampleException', 'AnotherException',
            'direct_import_function_as_an_argument', 'direct_import_second_function_as_an_argument', 
            'first_function_as_an_argument', 'second_function_as_an_argument',
            'direct_import_function_as_a_kwarg', 'direct_import_second_function_as_a_kwarg',
            'first_function_as_a_kwarg', 'second_function_as_a_kwarg'
        ],
        'count': [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)


def test_elements_counter(expected_dataframe):
    df = pd.DataFrame(columns=['filename', 'module', 'component_type', 'component_name', 'count'])
    tree = ast.parse(code)
    
    def walk_tree(node):
        check_node(node, test_lib_api['xyz'], df, 'test', 'xyz', direct_imports)
        for child in ast.iter_child_nodes(node):
            walk_tree(child)
    
    walk_tree(tree)

    assert_frame_equal(df.sort_values(by=['component_type', 'component_name']).reset_index(drop=True),
                       expected_dataframe.sort_values(by=['component_type', 'component_name']).reset_index(drop=True),
                       "DataFrame does not match the expected structure or counts")
