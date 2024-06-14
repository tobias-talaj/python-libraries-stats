import ast
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from lib_elements_counter import check_node, get_object_names


code = """
from test_lib import test_function_a, TestClassA, TestExceptionA
from test_lib import test_submodule
from test_lib.test_module import test_module_function_a, TestModuleClassA, TestModuleExceptionA


test_function_a()  # test_function_a 1
foo = test_function_a()  # test_function_a 2
bar = test_function_a  # NOPE
some_function(test_function_a)  # test_function_a 3
some_function(kwarg=test_function_a)  # test_function_a 4

test_lib.test_function_b()  # 1
foo = test_lib.test_function_b()  # 2
bar = test_lib.test_function_b  # NOPE
some_function(test_lib.test_function_b)  # test_function_b 3
some_function(kwarg=test_lib.test_function_b)  # test_function_b 4

some_object = TestClassA()  # TestClassA 1
some_object.test_attribute_a  # test_attribute_a 1
TestClassA().test_attribute_a  # TestClassA 2, test_attribute_a 2
some_other_object = some_object()
some_other_object.test_attribute_a  # test_attribute_a 3
another_object = some_object
another_object.test_attribute_a  # test_attribute_a 4
some_function(TestClassA())  # TestClassA 3

some_object = test_lib.TestClassB()  # TestClassB 1
some_object.test_attribute_b  # test_attribute_b 1
test_lib.TestClassB().test_attribute_b  # TestClassB 2, test_attribute_b 2
some_other_object = some_object()
some_other_object.test_attribute_b  # test_attribute_b 3
another_object = some_object
another_object.test_attribute_b  # test_attribute_b 4
some_function(test_lib.TestClassB())  # TestClassB 3

test_lib.TestClassC().test_method_a()  # TestClassC 1, test_method_a 1
some_object = test_lib.TestClassC()  # TestClassC 2
some_object.test_method_a()  # test_method_a 2

try:
    something()
except TestExceptionA:  # TestExceptionA 1
    pass 
try:
    something()
except test_lib.TestExceptionB:  # TestExceptionB 1
    pass
if some_condition:
    raise TestExceptionA()  # TestExceptionA 2
if some_condition:
    raise test_lib.TestExceptionB()  # TestExceptionB 2


test_module_function_a()  # test_module_function_a 1
foo = test_module_function_a()  # test_module_function_a 2
bar = test_module_function_a  # NOPE
some_function(test_module_function_a)  # test_module_function_a 3
some_function(kwarg=test_module_function_a)  # test_module_function_a 4

test_lib.test_module_function_b()  # 1
foo = test_lib.test_module_function_b()  # 2
bar = test_lib.test_module_function_b  # NOPE
some_function(test_lib.test_module_function_b)  # test_module_function_b 3
some_function(kwarg=test_lib.test_module_function_b)  # test_module_function_b 4

test_submodule.test_function_c()  # test_function_c 1

some_object = TestModuleClassA()  # TestModuleClassA 1
some_object.test_module_attribute_a  # test_module_attribute_a 1
TestModuleClassA().test_module_attribute_a  # TestModuleClassA 2, test_module_attribute_a 2
some_other_object = some_object()
some_other_object.test_module_attribute_a  # test_module_attribute_a 3
another_object = some_object
another_object.test_module_attribute_a  # test_module_attribute_a 4
some_function(TestModuleClassA())  # TestModuleClassA 3

some_object = test_lib.TestModuleClassB(not_important())  # TestModuleClassB 1
some_object.test_module_attribute_b  # test_module_attribute_b 1
test_lib.TestModuleClassB().test_module_attribute_b  # TestModuleClassB 2, test_module_attribute_b 2
some_other_object = some_object()
some_other_object.test_module_attribute_b  # test_module_attribute_b 3
another_object = some_object
another_object.test_module_attribute_b  # test_module_attribute_b 4
some_function(test_lib.TestModuleClassB())  # TestModuleClassB 3

test_lib.TestModuleClassC().test_module_method_a()  # TestModuleClassC 1, test_module_method_a 1
some_object = test_lib.TestModuleClassC()  # TestModuleClassC 2
some_object.test_module_method_a()  # test_module_method_a 2

try:
    something()
except TestModuleExceptionA:  # TestModuleExceptionA 1
    pass 
try:
    something()
except test_lib.TestModuleExceptionB:  # TestModuleExceptionB 1
    pass
if some_condition:
    raise TestModuleExceptionA()  # TestModuleExceptionA 2
if some_condition:
    raise test_lib.TestModuleExceptionB()  # TestModuleExceptionB 2
    
"""

test_lib_api = {
    'test_lib': {
        'function': ['test_function_a', 'test_function_b', 'test_function_c', 'test_module_function_a', 'test_module_function_b'],
        'method': ['test_method_a', 'test_method_b', 'test_module_method_a', 'test_module_method_b'],
        'class': ['TestClassA', 'TestClassB', 'TestClassC', 'TestModuleClassA', 'TestModuleClassB', 'TestModuleClassC'],
        'attribute': ['test_attribute_a', 'test_attribute_b', 'test_module_attribute_a', 'test_module_attribute_b'],
        'exception': ['TestExceptionA', 'TestExceptionB', 'TestModuleExceptionA', 'TestModuleExceptionB']
    }
}

direct_imports= {
    'test_lib': {
        'test_submodule',
        'test_function_a',
        'TestClassA',
        'TestExceptionA',
        'test_module_function_a',
        'TestModuleClassA',
        'TestModuleExceptionA'
    }
}

object_names = {'another_object', 'some_object', 'some_other_object'}

@pytest.fixture
def expected_dataframe():
    data = {
        "filename": ["test"] * 21,
        "module": ["test_lib"] * 21,
        "component_type": [
            "function", "function", "function", "class", "attribute", 
            "class", "attribute", "method", "class", "exception", 
            "exception", "function", "function", "class", "attribute",
            "class", "attribute", "method", "class", "exception",
            "exception"
        ],
        "component_name": [
            "test_function_a", "test_function_b", "test_function_c", "TestClassA", "test_attribute_a", 
            "TestClassB", "test_attribute_b", "test_method_a", "TestClassC", "TestExceptionA", 
            "TestExceptionB", "test_module_function_a", "test_module_function_b", "TestModuleClassA", "test_module_attribute_a",
            "TestModuleClassB", "test_module_attribute_b", "test_module_method_a", "TestModuleClassC", "TestModuleExceptionA",
            "TestModuleExceptionB"
        ],
        "count": [
            4, 4, 1, 3, 4, 
            3, 4, 2, 2, 2, 
            2, 4, 4, 3, 4,
            3, 4, 2, 2, 2,
            2
        ]
    }
    return pd.DataFrame(data)

def test_elements_counter(expected_dataframe):
    df = pd.DataFrame(columns=['filename', 'module', 'component_type', 'component_name', 'count'])
    tree = ast.parse(code)
    
    def walk_tree(node):
        check_node(node, test_lib_api['test_lib'], df, 'test', 'test_lib', direct_imports['test_lib'], object_names)
        for child in ast.iter_child_nodes(node):
            walk_tree(child)
    
    walk_tree(tree)

    assert_frame_equal(df.sort_values(by=['component_type', 'component_name']).reset_index(drop=True),
                       expected_dataframe.sort_values(by=['component_type', 'component_name']).reset_index(drop=True),
                       "DataFrame does not match the expected structure or counts")
