import pytest

def terry(x):
    return x + 5

def test_method():
    assert terry(3) == 8
