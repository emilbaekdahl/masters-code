def delegate(*methods, to):
    """Delegates one or more methods to an attribute of a class.

    Parameters
        methods: Names of the methods to delegate.
        to: object that the relavante methods will be delegated to.

    Returns:
        Decorated class.

    Exampels:
        >>> @delagate("sum", "mean", to="aggregator")
        >>> class Collection:
        >>>     def __init__(self, values):
        >>>         self.values = values
        >>>         self.aggregator = Aggregator(self)
        >>>
        >>> class Aggregator:
        >>>     def __init__(self, collection):
        >>>         self.collection = collection
        >>>
        >>>     def sum(self):
        >>>         return sum(self.collection.values)
        >>>
        >>>     def mean(self):
        >>>         return self.sum() / len(self.collection.values)
        >>>
        >>> collection = Collection([1, 10, 100, 1000])
        >>> collection.sum()
        1111
        >>> collection.mean()
        227,75
    """

    def define_method(name):
        def temp(self, *args, **kwargs):
            to_object = getattr(self, to)
            to_method = getattr(to_object, name)

            return to_method(*args, **kwargs)

        temp.__name__ = name
        return temp

    def _delegate(klass):
        for method in methods:
            setattr(klass, method, define_method(method))

        return klass

    return _delegate
