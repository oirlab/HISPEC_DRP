
class _FiberProxy:
    """
    Proxy returned by ``model.spec_fib`` (any attr ending in ``_fib``).

    Supports subscript get/set so that ``model.spec_fib[fiber]`` is
    equivalent to ``model.spec_fib{fiber}``.
    """

    __slots__ = ('_model', '_base')

    def __init__(self, model, base: str):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_base', base)

    def __getitem__(self, fiber: str | int):
        return getattr(self._model, f'{self._base}{fiber}')

    def __setitem__(self, fiber: str | int, value):
        setattr(self._model, f'{self._base}{fiber}', value)

    def __repr__(self):
        return f'<_FiberProxy {self._base!r}>'


class _FiberView:
    """
    Proxy returned by ``model[fiber]``.

    Allows accessing any fiber-indexed extension without the ``_fib`` suffix,
    e.g. ``model[1].spec`` is equivalent to ``model.spec_fib1``.
    """

    __slots__ = ('_model', '_fiber')

    def __init__(self, model, fiber: str | int):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_fiber', fiber)

    def __getattr__(self, name: str):
        return getattr(self._model, f'{name}_fib{self._fiber}')

    def __setattr__(self, name: str, value):
        setattr(self._model, f'{name}_fib{self._fiber}', value)

    def __repr__(self):
        return f'<_FiberView fiber={self._fiber!r}>'


class FiberAttributeProxyMixin:
    """
    Mixin that adds fiber-indexed attribute proxy access to a DataModel.

    Features
    --------
    - ``model.spec_fib[fiber]``  →  ``model.spec_fib{fiber}``  (get)
    - ``model.spec_fib[fiber] = value``  →  ``model.spec_fib{fiber} = value``  (set)
    - ``model[fiber].spec``  →  ``model.spec_fib{fiber}``  (get, via fiber view)
    - ``model[fiber].spec = value``  →  ``model.spec_fib{fiber} = value``  (set, via fiber view)

    *fiber* may be an int or str, e.g. ``1`` or ``'1'``.

    Usage
    -----
    Place the mixin **before** the base DataModel in the class definition so
    that its ``__getattr__`` and ``__setattr__`` hooks are resolved first::

        class L2Model(FiberAttributeProxyMixin, DataModel): ...
    """

    def __getattr__(self, attr: str):
        attrl = attr.lower()
        # model.spec_fib[fiber] — attr already carries the _fib suffix, no trailing digit.
        if attrl.endswith('_fib'):
            return _FiberProxy(self, attrl)
        # model.spec[fiber] — short form: check whether attr + '_fib' is a valid
        # extension base (i.e. EXTENSIONS contains at least one ATTR_FIB# key).
        try:
            extensions = object.__getattribute__(self, 'EXTENSIONS')
        except AttributeError:
            extensions = {}
        if any(k.startswith(attr.upper() + '_FIB') for k in extensions):
            return _FiberProxy(self, attrl + '_fib')
        return super().__getattr__(attr)

    def __setattr__(self, attr: str, value):
        # Pass through to DataModel.__setattr__ which validates against EXTENSIONS.
        return super().__setattr__(attr, value)

    def __getitem__(self, fiber: str | int):
        # model[fiber] returns a view that maps attribute names to _fib{fiber} attrs.
        return _FiberView(self, fiber)

    def __setitem__(self, fiber: str | int, value):
        raise TypeError(
            f"Cannot assign to model[{fiber!r}] directly. "
            "Use model.<ext>_fib[fiber] = value, e.g. model.spec_fib[1] = arr."
        )