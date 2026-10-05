import logging

from medellin_rent.utils.log import get_logger


def test_get_logger_does_not_duplicate_handlers() -> None:
    first = get_logger("medellin_rent.test", "DEBUG")
    second = get_logger("medellin_rent.test")
    assert first is second
    assert len(second.handlers) == 1
    assert second.level == logging.INFO
