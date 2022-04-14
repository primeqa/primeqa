def hello_world() -> str:
    """
    Returns:
        'Hello, World!'
    """
    return "Hello, World!"


def hello_name(name: str) -> str:
    """
    Args:
        name: say hello to

    Returns:
        f"Hello, {name}!"
    """
    return f"Hello, {name}!"


class HelloWorldException(NotImplementedError):
    pass


class HelloWorld:
    def __init__(self, name: str):
        self._name = name

    def to_str(self) -> str:
        """
        Returns:
            str rep of object
        """
        return str(self)

    def foo(self) -> int:
        """
        Returns:
            1
        """
        return 1

    def bar(self) -> list:
        """
        Returns:
            [1, 2, 3]
        """
        return [1, 2, 3]

    def baz(self) -> dict:
        """
        Returns:
            dict of a-c -> 1-3
        """
        return {'a': 1, 'b': 2, 'c': 3}

    def __str__(self):
        return self._name

    def __iter__(self):
        raise HelloWorldException
