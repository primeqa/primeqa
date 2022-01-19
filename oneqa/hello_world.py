def hello_world() -> str:
    """
    Hello, world!

    :return: "Hello, world!"
    """
    return "Hello, World!"


def hello_name(name: str) -> str:
    """
    Hello, {name}!

    :param name: who to say hello to
    :return: "Hello, {name}!"
    """
    return f"Hello, {name}!"


class HelloWorldException(NotImplementedError):
    pass


class HelloWorld:
    def __init__(self, name: str):
        self._name = name

    def to_str(self) -> str:
        """
        :return: str representation of object
        """
        return str(self)

    def foo(self) -> int:
        """
        :return: 1
        """
        return 1

    def bar(self) -> list:
        """
        :return: [1, 2, 3]
        """
        return [1, 2, 3]

    def baz(self) -> dict:
        """
        :return: {'a': 1, 'b': 2, 'c': 3}
        """
        return {'a': 1, 'b': 2, 'c': 3}

    def __str__(self):
        return self._name

    def __iter__(self):
        raise HelloWorldException
