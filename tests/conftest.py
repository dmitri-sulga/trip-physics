"""Shared pytest configuration for the test suite.

Registers the ``slow`` marker locally rather than in pyproject.toml, so the
packaging metadata stays owned by the build and the test suite owns its own
collection rules.

``slow`` marks tests that score a 20000-account corpus. They are part of the
default run because they anchor the numbers the harness actually quotes, but
they can be deselected during rapid iteration:

    pytest -m "not slow"
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: scores a 20k-account corpus; deselect with -m 'not slow'",
    )
