from datetime import datetime, timedelta, timezone

from roboatom.annotation.app import format_http_datetime


def test_format_http_datetime_normalizes_naive_datetime_to_utc() -> None:
    assert format_http_datetime(datetime(2026, 8, 11, 8, 30, 0)) == (
        "Tue, 11 Aug 2026 08:30:00 GMT"
    )


def test_format_http_datetime_converts_nonstandard_timezone_to_utc() -> None:
    oss_timezone = timezone(timedelta(hours=8), name="OSS-CST")
    assert format_http_datetime(datetime(2026, 8, 11, 16, 30, 0, tzinfo=oss_timezone)) == (
        "Tue, 11 Aug 2026 08:30:00 GMT"
    )
