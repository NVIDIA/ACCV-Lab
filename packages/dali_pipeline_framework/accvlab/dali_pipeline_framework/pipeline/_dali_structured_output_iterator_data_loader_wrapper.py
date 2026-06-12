# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dali_structured_output_iterator import DALIStructuredOutputIterator


def get_masked_as_type(to_set_class, to_set_as_type):
    class ClassTypeAsClassTypeWrapper(to_set_class, to_set_as_type):

        _ALLOWED_MAGIC = frozenset(
            ("__class__", "__dict__", "__weakref__", "__module__", "__doc__", "__annotations__")
        )

        def __init__(self, *args, **kwargs):
            # Intentionally do not initialize the masked parent type
            to_set_class.__init__(self, *args, **kwargs)

        def __getattribute__(self, name):
            # Allow basic introspection attributes
            if name in type(self)._ALLOWED_MAGIC:
                return object.__getattribute__(self, name)

            # Always allow direct instance attributes
            instance_dict = object.__getattribute__(self, "__dict__")
            if name in instance_dict:
                return instance_dict[name]

            # Determine which class in the MRO first defines this attribute
            for cls in type(self).__mro__:
                if name in cls.__dict__:
                    # Block attributes originating from the masked parent type lineage
                    if cls is not to_set_class and issubclass(cls, to_set_as_type):
                        masked_type_name = getattr(to_set_as_type, "__qualname__", to_set_as_type.__name__)
                        masked_type_mod = getattr(
                            to_set_as_type, "__module__", to_set_as_type.__class__.__module__
                        )
                        raise RuntimeError(
                            f"Access to attribute '{name}' is disabled because it originates from "
                            f"{masked_type_mod}.{masked_type_name} on {type(self).__name__}."
                        )
                    break

            return object.__getattribute__(self, name)

        # Always allow setting attributes, but make sure that if __setattr__ is not defined in the left
        # parent, we bypass the right parent's setter.
        def __setattr__(self, name, value):
            # Delegate to left parent if it defines __setattr__, otherwise bypass masked parent's setter
            if "__setattr__" in to_set_class.__dict__:
                return to_set_class.__setattr__(self, name, value)
            return object.__setattr__(self, name, value)

        # Always allow deleting attributes, but make sure that if __delattr__ is not defined in the left
        # parent, we bypass the right parent's deleter.
        def __delattr__(self, name):
            # Delegate to left parent if it defines __delattr__, otherwise bypass masked parent's deleter
            if "__delattr__" in to_set_class.__dict__:
                return to_set_class.__delattr__(self, name)
            return object.__delattr__(self, name)

    # Dynamically mask special methods defined by the masked parent type that
    # are not implemented by DALIStructuredOutputIterator and not already overridden above.
    wrapper_cls = ClassTypeAsClassTypeWrapper
    skip_names = set(wrapper_cls._ALLOWED_MAGIC) | {
        "__getattribute__",
        "__init__",
        "__new__",
        "__class__",
        "__mro_entries__",
        "__init_subclass__",
        "__subclasshook__",
        "__getattr__",
        "__get__",
        "__set__",
        "__delete__",
        "__prepare__",
        "__class_getitem__",
        "__setattr__",
        "__delattr__",
    }

    def _make_raiser(name):
        def _raiser(self, *args, **kwargs):
            masked_type_name = getattr(to_set_as_type, "__qualname__", to_set_as_type.__name__)
            masked_type_mod = getattr(to_set_as_type, "__module__", to_set_as_type.__class__.__module__)
            raise RuntimeError(
                f"Access to special method '{name}' is disabled because it originates from "
                f"{masked_type_mod}.{masked_type_name} on {type(self).__name__}."
            )

        _raiser.__name__ = name
        return _raiser

    for _attr_name, _attr_val in to_set_as_type.__dict__.items():
        if not (_attr_name.startswith("__") and _attr_name.endswith("__")):
            continue
        if _attr_name in skip_names:
            continue
        # Do not mask if DALIStructuredOutputIterator provides its own implementation
        if _attr_name in to_set_class.__dict__:
            continue
        # Do not mask if we already provided a specific override above
        if _attr_name in wrapper_cls.__dict__:
            continue
        setattr(wrapper_cls, _attr_name, _make_raiser(_attr_name))

    return wrapper_cls
