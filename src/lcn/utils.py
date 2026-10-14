# Copyright (c) 2025, Lawrence Livermore National Security, LLC. All rights reserved. LLNL-CODE-2025335
# This work was produced at the Lawrence Livermore National Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44)
# between the U.S. Department of Energy (DOE) and Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
# See license for disclaimers, notice of U.S. Government Rights and license terms and conditions.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Iterable, List, TypeVar

T = TypeVar("T")

def intersection(lst1: Iterable[T], lst2: Iterable[T]) -> List[T]:
    """Return elements of lst1 that are also in lst2, preserving order."""
    set2 = set(lst2)
    return [value for value in lst1 if value in set2]
