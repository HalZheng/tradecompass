"""交易日历（SPEC §8.7）。

只暴露日历相关能力；上交所日历（XSHG）与深市、北交所的交易日一致。
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

CALENDAR_NAME = "XSHG"


class CalendarError(ValueError):
    """日期超出日历覆盖范围，或不是合法交易日。"""


@lru_cache(maxsize=1)
def _calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar(CALENDAR_NAME)


class TradingCalendar:
    """交易日判定与前后交易日查找。"""

    def is_trading_day(self, day: date) -> bool:
        cal = _calendar()
        ts = pd.Timestamp(day)
        if ts < cal.first_session or ts > cal.last_session:
            raise CalendarError(f"{day} 超出日历覆盖范围 {cal.first_session}~{cal.last_session}")
        return bool(cal.is_session(ts))

    def previous_trading_day(self, day: date) -> date:
        cal = _calendar()
        session = cal.date_to_session(pd.Timestamp(day), direction="previous")
        result: date = pd.Timestamp(session).date()
        return result

    def next_trading_day(self, day: date) -> date:
        cal = _calendar()
        session = cal.date_to_session(pd.Timestamp(day), direction="next")
        result: date = pd.Timestamp(session).date()
        return result

    def trading_days(self, start: date, end: date) -> list[date]:
        if end < start:
            raise CalendarError(f"end({end}) 早于 start({start})")
        cal = _calendar()
        sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [pd.Timestamp(item).date() for item in sessions]

    def sessions(self, start: date, count: int) -> list[date]:
        """从 ``start``（含）起最多 ``count`` 个交易日。

        日历只覆盖到某一天，未来日期拿不满时返回能取到的部分，不抛异常——
        回填建议表现时“数据还没到”是常态，不该当成错误。
        """
        if count <= 0:
            return []
        cal = _calendar()
        ts = pd.Timestamp(start)
        if ts < cal.first_session or ts > cal.last_session:
            return []
        sessions = cal.sessions_in_range(ts, cal.last_session)
        return [pd.Timestamp(item).date() for item in sessions[:count]]

    def last_trading_day_on_or_before(self, day: date) -> date:
        """含当日：当日是交易日则返回当日，否则返回上一个交易日。"""
        return self.previous_trading_day(day)


@lru_cache(maxsize=1)
def default_calendar() -> TradingCalendar:
    return TradingCalendar()
