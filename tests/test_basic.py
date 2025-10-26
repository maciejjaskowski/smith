import smith


def test_version_and_greet():
    assert isinstance(smith.__version__, str)
    assert smith.greet("World") == "Hello, World"
