"""Shared pytest policy for the authoritative verification gate."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--max-skips",
        type=int,
        default=None,
        help="Fail the session when skipped tests exceed this count",
    )


def pytest_sessionfinish(session, exitstatus):
    maximum = session.config.getoption("--max-skips")
    if maximum is None:
        return
    # TerminalReporter owns the authoritative outcome counts.
    reporter = session.config.pluginmanager.get_plugin(
        "terminalreporter")
    skipped = len(reporter.stats.get("skipped", [])) if reporter else 0
    if skipped > maximum and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        if reporter:
            reporter.write_line(
                f"skip budget exceeded: {skipped} > {maximum}",
                red=True,
            )
