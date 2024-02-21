"""PyTorch container implementations to fix buffer registration and typing.

Members are explicitly re-exported in pyprobound.
"""

from __future__ import annotations

import collections
import operator
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
)
from typing import TypeVar, cast, overload

import torch
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import Self, override

from . import __version__

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


class TModuleList(torch.nn.Module, MutableSequence[ModuleT]):
    """Typed ModuleList.

    See https://github.com/pytorch/pytorch/pull/89135.
    """

    _modules: collections.OrderedDict[str, ModuleT]  # type: ignore[assignment]

    def __init__(self, modules: Iterable[ModuleT] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, index: int) -> str:
        """Get the absolute index for the list of modules."""
        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range")
        if index < 0:
            index += len(self)
        return str(index)

    @overload
    def __getitem__(self, index: int) -> ModuleT: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @override
    def __getitem__(self, index: int | slice) -> ModuleT | Self:
        if isinstance(index, slice):
            return self.__class__(list(self._modules.values())[index])
        return self._modules[self._get_abs_string_index(index)]

    @overload
    def __setitem__(self, index: int, value: ModuleT) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ModuleT]) -> None: ...

    @override
    def __setitem__(
        self, index: int | slice, value: ModuleT | Iterable[ModuleT]
    ) -> None:
        if isinstance(index, slice) and isinstance(value, Iterable):
            start, stop, step = index.indices(len(self))
            iteration = range(start, stop, step)
            for idx, val in zip(iteration, value):
                str_idx = self._get_abs_string_index(idx)
                if not isinstance(val, torch.nn.Module):
                    raise TypeError(
                        "Expected a Module for TModulelist"
                        f" but received {type(val).__name__} instead"
                    )
                setattr(self, str_idx, val)
        elif isinstance(index, int):
            str_idx = self._get_abs_string_index(index)
            if not isinstance(value, torch.nn.Module):
                raise TypeError(
                    "Expected a Module for TModuleList"
                    f" but received {type(value).__name__} instead"
                )
            setattr(self, str_idx, value)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    @override
    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            for k in range(len(self._modules))[index]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(index))
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = collections.OrderedDict(
            zip(str_indices, self._modules.values())  # preserve numbering
        )

    @override
    def __len__(self) -> int:
        return len(self._modules)

    @override
    def insert(self, index: int, value: ModuleT) -> None:
        """Insert value before index."""
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = value

    @override
    def __hash__(self) -> int:
        return torch.nn.Module.__hash__(self)


class TParameterList(torch.nn.Module, MutableSequence[Parameter]):
    """Typed ParameterList.

    See https://github.com/pytorch/pytorch/pull/89135.
    """

    _parameters: collections.OrderedDict[str, Parameter]  # type: ignore[assignment]

    def __init__(self, values: Iterable[Tensor] | None = None) -> None:
        super().__init__()
        if values is not None:
            params = [
                Parameter(i) if not isinstance(i, Parameter) else i
                for i in values
            ]
            self += params

    def _get_abs_string_index(self, index: int) -> str:
        """Get the absolute index for the list of modules"""
        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range")
        if index < 0:
            index += len(self)
        return str(index)

    @overload
    def __getitem__(self, index: int) -> Parameter: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @override
    def __getitem__(self, index: int | slice) -> Parameter | Self:
        if isinstance(index, slice):
            return self.__class__(list(self._parameters.values())[index])
        return self._parameters[self._get_abs_string_index(index)]

    @overload
    def __setitem__(self, index: int, value: Tensor) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Tensor]) -> None: ...

    @override
    def __setitem__(
        self, index: int | slice, value: Tensor | Iterable[Tensor]
    ) -> None:
        if isinstance(index, slice) and isinstance(value, Iterable):
            start, stop, step = index.indices(len(self))
            iteration = range(start, stop, step)
            for idx, val in zip(iteration, value):
                str_idx = self._get_abs_string_index(idx)
                val = Parameter(val) if not isinstance(val, Parameter) else val
                if not isinstance(val, Tensor):
                    raise TypeError(
                        "Expected a Tensor for TParameterList"
                        f" but received {type(val).__name__} instead"
                    )
                setattr(self, str_idx, val)
        elif isinstance(index, int) and isinstance(value, Tensor):
            str_idx = self._get_abs_string_index(index)
            if not isinstance(value, Parameter):
                param = Parameter(value)
            else:
                param = value
            if not isinstance(value, Tensor):
                raise TypeError(
                    "Expected a Tensor for TParameterList"
                    f" but received {type(value).__name__} instead"
                )
            setattr(self, str_idx, param)
        else:
            raise TypeError(
                f"index type {type(index).__name__}"
                f" incompatible with value type {type(value).__name__}"
            )

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    @override
    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            for k in range(len(self._parameters))[index]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(index))
        str_indices = [str(i) for i in range(len(self._parameters))]
        self._parameters = collections.OrderedDict(
            zip(str_indices, self._parameters.values())  # preserve numbering
        )

    @override
    def __len__(self) -> int:
        return len(self._parameters)

    @override
    def insert(self, index: int, value: Tensor) -> None:
        """Insert value before index."""
        for i in range(len(self._parameters), index, -1):
            self._parameters[str(i)] = self._parameters[str(i - 1)]
        param = Parameter(value) if not isinstance(value, Parameter) else value
        self._parameters[str(index)] = param

    @override
    def __hash__(self) -> int:
        return torch.nn.Module.__hash__(self)


class TParameterDict(torch.nn.Module, MutableMapping[str, Parameter]):
    """Typed ParameterDict.

    See https://github.com/pytorch/pytorch/pull/92032.
    """

    _parameters: collections.OrderedDict[str, Parameter]  # type: ignore[assignment]

    def __init__(
        self,
        parameters: (
            Mapping[str, Parameter] | Iterable[tuple[str, Parameter]] | None
        ) = None,
        **kwargs: Parameter,
    ) -> None:
        super().__init__()
        self._keys: set[str] = set()
        if parameters is not None:
            self.update(parameters)
        if len(kwargs) > 0:
            self.update(**kwargs)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(
                "Index given to TParameterDict cannot be used as key"
                f" as it is not a string (type is '{type(key).__name__}')."
            )
        return key

    @override
    def __getitem__(self, __key: str) -> Parameter:
        if __key not in self._keys:
            raise KeyError(__key)
        attr = self._key_to_attr(__key)
        return cast(Parameter, getattr(self, attr))

    @override
    def __setitem__(self, __key: str, __value: Parameter) -> None:
        attr = self._key_to_attr(__key)
        self._keys.add(attr)
        if not isinstance(__value, torch.nn.Parameter):
            raise TypeError(
                "Expected a parameter for TParameterDict"
                f" but received {type(__value).__name__} instead"
            )
        setattr(self, attr, __value)

    @override
    def __delitem__(self, __key: str) -> None:
        self._keys.remove(__key)
        attr = self._key_to_attr(__key)
        delattr(self, attr)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    @override
    def __len__(self) -> int:
        return len(self._keys)

    @override
    def __hash__(self) -> int:
        return torch.nn.Module.__hash__(self)
