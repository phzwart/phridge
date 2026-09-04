"""Import cctbx before torch: importing torch first segfaults cctbx's Boost.Python extensions."""

try:
    import cctbx  # noqa: F401
    from cctbx import xray  # noqa: F401
except ImportError:
    pass
