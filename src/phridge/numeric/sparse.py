"""Sparse assembly and factorisation."""

from __future__ import annotations

from typing import Any


def csr_matrix(data: Any, shape: tuple[int, int]) -> Any:
    import scipy.sparse as sp

    return sp.csr_matrix(data, shape=shape)


def diags(values: Any) -> Any:
    import scipy.sparse as sp

    return sp.diags(values)


def cholmod_factor(matrix: Any) -> Any:
    from sksparse.cholmod import cholesky

    return cholesky(matrix.tocsc())


def splu(matrix: Any) -> Any:
    import scipy.sparse.linalg as spl

    return spl.splu(matrix.tocsc())
