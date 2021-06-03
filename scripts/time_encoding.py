import datetime
from enum import Enum

from pandas import DataFrame

import dateparser
import holidays
from dateutil.parser import parse


class TimeType(Enum):
    DATE = 'date'
    DURATION = 'duration'
    NONE = 'none'


def main():
    time_test = [
        '189161464',
        '1990-12-1',
        '2005/3',
        'Jan 19, 1990',
        'Monday at 12:01am',
        '01/19/1990',
        '01/19/90',
        'Jan 1990',
        'January1990',
        'January 1, 2047 at 8:21:00AM'
    ]

    duration_test = [
        '161613484',
        '2d9h32m46s',
        '2d 9h 37m 46s',
        '2days9hours37minutes46seconds',
        '2days 9hours 37minutes 46seconds'
    ]

    print(is_time_or_duration(time_test))
    print(is_time_or_duration(duration_test))


def is_time_or_duration(column: list):
    """Returns whether the column contains dates, durations, or otherwise

    :param column:
    :return:
    """
    column_type = ''
    if is_duration(column):
        column_type = TimeType.DURATION.value
    elif is_date(column):
        column_type = TimeType.DATE.value
    else:
        column_type = TimeType.NONE.value

    return column_type


def is_date(column: list) -> bool:
    """Return whether all string can be interpreted as a date.

    Function take from https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    :param column: list of str, strings to check for date
    :return: True if all string of column are dates
    """
    is_all_dates = True

    for string in column:
        if string != "":
            try:
                dateparser.parse(string)
            except ValueError:
                is_all_dates = False

    return is_all_dates


def is_duration(column: list) -> bool:
    """Return whether all string can be interpreted as a duration.

    :param column: list of str, strings to check for periods of time
    :return: True if all string of column are periods of time
    """
    is_all_duration = False
    allowed_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    allowed_word = ['d', 'days', 'h', 'hours', 'm', 'minutes', 's', 'seconds']

    for string in column:
        string = string.replace(" ", "")
        words = string.split(":")
        if all(word in allowed_word or word in allowed_number for word in words)
    return is_all_duration


def parse_date(column: list) -> DataFrame:
    """Parses strings of column into datetime objects and returns a DataFrame

    :param column: list of str, strings to parse into date
    :return:
    """
    results_df = DataFrame(columns=['date_day', 'date_month', 'date_year', 'date_hours', 'date_minutes', 'date_seconds',
                                    'date_special_occasion'])
    for i in range(len(column)):
        date = dateparser.parse(column[i])  # Returns a datetime type
        row = [date.day, date.month, date.year, date.hour, date.minute, date.second]

        holiday = holidays.CountryHoliday('US')
        if date.strftime("%m-%d-%Y") in holiday:
            row.append(True)
        else:
            row.append(False)

        results_df.loc[i](row)

    return results_df


def parse_duration(column: list) -> DataFrame:
    """Parses strings of column into datetime objects and returns a DataFrame

    :param column:
    :return:
    """
    results_df = DataFrame(columns=['date_days', 'date_hours', 'date_minutes', 'date_seconds'])

    for i in range(len(column)):
        row = []

        # TODO: complete

        results_df.loc[i](row)

    return results_df


main()
