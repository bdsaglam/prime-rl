"""Tests for the ARC-AGI REPL environment."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_agi.envs.repl import ARC_SANDBOX_CODE, parse_response
from arc_agi.subprocess_interpreter import SubprocessInterpreter


def test_parse_structured_response():
    """Test parsing a well-formed structured response."""
    text = """
[[ ## reasoning ## ]]
This is my reasoning about the problem.
I think the pattern is rotation.

[[ ## code ## ]]
```python
print("hello")
x = 1 + 2
```
"""
    sections = parse_response(text)
    assert "rotation" in sections["reasoning"], f"Expected 'rotation' in reasoning: {sections['reasoning']}"
    assert 'print("hello")' in sections["code"], f"Expected print in code: {sections['code']}"
    assert "x = 1 + 2" in sections["code"], f"Expected 'x = 1 + 2' in code: {sections['code']}"
    print("✓ test_parse_structured_response passed")


def test_parse_code_without_fences():
    """Test parsing code section without markdown fences."""
    text = """
[[ ## reasoning ## ]]
Simple test.

[[ ## code ## ]]
x = 42
print(x)
"""
    sections = parse_response(text)
    assert "x = 42" in sections["code"], f"Expected 'x = 42' in code: {sections['code']}"
    print("✓ test_parse_code_without_fences passed")


def test_fallback_to_code_fences():
    """Test fallback extraction when no section markers present."""
    text = """
Here's some code:
```python
result = 1 + 1
```
"""
    sections = parse_response(text)
    assert "result = 1 + 1" in sections["code"], f"Expected code in sections: {sections['code']}"
    print("✓ test_fallback_to_code_fences passed")


def test_multiple_code_sections():
    """Test that multiple [[ ## code ## ]] sections are concatenated."""
    text = """
[[ ## reasoning ## ]]
First compute the output, then submit.

[[ ## code ## ]]
```python
output = np.array([[1, 2], [3, 4]])
print(format_grid(output))
```

[[ ## code ## ]]
```python
SUBMIT(test=[output])
```
"""
    sections = parse_response(text)
    assert "output = np.array" in sections["code"], f"Expected first code block: {sections['code']}"
    assert "SUBMIT(test=[output])" in sections["code"], f"Expected second code block: {sections['code']}"
    # Verify order: first block before second
    idx1 = sections["code"].index("output = np.array")
    idx2 = sections["code"].index("SUBMIT")
    assert idx1 < idx2, "First code block should come before second"
    print("✓ test_multiple_code_sections passed")


def test_multiple_code_fences_fallback():
    """Test that multiple code fences in fallback mode are concatenated."""
    text = """
Here is some code:
```python
x = 1
```

And more:
```python
y = x + 1
```
"""
    sections = parse_response(text)
    assert "x = 1" in sections["code"], f"Expected first block: {sections['code']}"
    assert "y = x + 1" in sections["code"], f"Expected second block: {sections['code']}"
    print("✓ test_multiple_code_fences_fallback passed")


def test_empty_response():
    """Test handling of empty response."""
    sections = parse_response("")
    assert sections["reasoning"] == "", f"Expected empty reasoning: {sections['reasoning']}"
    assert sections["code"] == "", f"Expected empty code: {sections['code']}"
    print("✓ test_empty_response passed")


def test_task_variable_injection():
    """Test that task data is correctly injected and accessible."""
    # Sample task data
    task_data = {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
        ],
        "test": [
            {"input": [[4, 5], [5, 4]]},
        ],
    }

    # Format sandbox code
    sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))

    # Create interpreter and execute
    interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)
    try:
        interpreter.start()

        # Test that task variable exists and has correct structure
        result = interpreter.execute("print(len(task['train']))")
        assert "2" in result, f"Expected 2 train examples: {result}"

        result = interpreter.execute("print(len(task['test']))")
        assert "1" in result, f"Expected 1 test example: {result}"

        # Test train data structure
        result = interpreter.execute("print(task['train'][0]['input'])")
        assert "[0, 1]" in result, f"Expected train input: {result}"

        result = interpreter.execute("print(task['train'][0]['output'])")
        assert "[1, 0]" in result, f"Expected train output: {result}"

        # Test test data structure (no output key)
        result = interpreter.execute("print(task['test'][0]['input'])")
        assert "[4, 5]" in result, f"Expected test input: {result}"

        result = interpreter.execute("print('output' in task['test'][0])")
        assert "False" in result, f"Expected no output in test: {result}"

        print("✓ test_task_variable_injection passed")
    finally:
        interpreter.shutdown()


def test_helper_functions_available():
    """Test that helper functions are available in sandbox."""
    task_data = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }

    sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))
    interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)

    try:
        interpreter.start()

        # Test format_grid
        result = interpreter.execute("print(format_grid([[1, 2], [3, 4]]))")
        assert "1 2" in result, f"Expected formatted grid: {result}"
        assert "3 4" in result, f"Expected formatted grid: {result}"

        # Test accuracy
        result = interpreter.execute("print(accuracy([[1, 2], [3, 4]], [[1, 2], [3, 4]]))")
        assert "1.0" in result, f"Expected accuracy 1.0: {result}"

        result = interpreter.execute("print(accuracy([[1, 2], [3, 4]], [[0, 0], [0, 0]]))")
        assert "0.0" in result, f"Expected accuracy 0.0: {result}"

        # Test soft_accuracy
        result = interpreter.execute("print(soft_accuracy([[1, 2], [3, 4]], [[1, 2], [0, 0]]))")
        assert "0.5" in result, f"Expected soft_accuracy 0.5: {result}"

        # Test numpy is available
        result = interpreter.execute("print(np.array([[1, 2], [3, 4]]).shape)")
        assert "(2, 2)" in result, f"Expected shape (2, 2): {result}"

        print("✓ test_helper_functions_available passed")
    finally:
        interpreter.shutdown()


def test_grid_manipulation():
    """Test typical grid manipulation workflow."""
    task_data = {
        "train": [{"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[0, 0, 0], [0, 0, 0]]}],
    }

    sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))
    interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)

    try:
        interpreter.start()

        # Simulate typical workflow
        interpreter.execute("pred = np.array(task['test'][0]['input']).copy()")
        interpreter.execute("pred[:, :] = 1")
        result = interpreter.execute("print(pred.tolist())")
        assert "[[1, 1, 1], [1, 1, 1]]" in result, f"Expected filled grid: {result}"

        # Verify against training
        interpreter.execute("train_pred = np.array(task['train'][0]['input']).copy()")
        interpreter.execute("train_pred[:, :] = 1")
        result = interpreter.execute("print(accuracy(train_pred, task['train'][0]['output']))")
        assert "1.0" in result, f"Expected accuracy 1.0: {result}"

        print("✓ test_grid_manipulation passed")
    finally:
        interpreter.shutdown()


if __name__ == "__main__":
    print("Running REPL environment tests...\n")

    # Parse response tests
    test_parse_structured_response()
    test_parse_code_without_fences()
    test_fallback_to_code_fences()
    test_multiple_code_sections()
    test_multiple_code_fences_fallback()
    test_empty_response()

    # Sandbox injection tests
    test_task_variable_injection()
    test_helper_functions_available()
    test_grid_manipulation()

    print("\n✓ All tests passed!")
