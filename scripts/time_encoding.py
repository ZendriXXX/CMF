import datetime
from enum import Enum

from pandas import *

from dateutil.parser import parse
import dateparser
import holidays


duration_allowed_word = ['d', 'days', 'h', 'hours', 'm', 'minutes', 's', 'seconds', 'ms']


class TimeType(Enum):
    DATE = 'date'
    DURATION = 'duration'
    NONE = 'none'


def main():
    time_test = [
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

    print(parse_date(time_test).head())
    print(parse_duration(duration_test).head())


def is_time_or_duration(column: list):
    """Returns whether the column contains dates, durations, or otherwise

    :param column:
    :return:
    """
    column_type = TimeType.NONE.value

    if is_duration(column):
        column_type = TimeType.DURATION.value
    elif is_date(column):
        column_type = TimeType.DATE.value

    return column_type


def is_date(column: list) -> bool:
    """Return whether all string can be interpreted as a date.

    Function take from https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    :param column: list of str, strings to check for date
    :return: True if all string of column are dates
    """
    is_all_dates = True

    for string in column:
        try:
            parse(string)
        except ValueError:
            is_all_dates = False

    return is_all_dates


def is_duration(column: list) -> bool:
    """Return whether all string can be interpreted as a duration.

    :param column: list of str, strings to check for periods of time
    :return: True if all string of column are periods of time
    """
    is_all_duration = True

    for string in column:
        if string != "":
            words = format_string_duration_parse(string)
            if not all(len(word) == 2 and word[1] in duration_allowed_word for word in words):
                is_all_duration = False

    return is_all_duration


def format_string_duration_parse(string: str) -> list:
    """Returns a list containing the given string split

    :param string:
    :return:
    """
    string = string.replace(" ", "")
    formatted_string = []

    if string.isnumeric():
        formatted_string.append((string, duration_allowed_word[8]))  # formatted_string.append((string, 'MS'))
    else:
        chars = [char for char in string]
        i = 0
        while i < (len(chars)-1):
            if not chars[i].isnumeric() and chars[i+1].isnumeric():
                chars.insert(i+1, "|")
                i += 2
            elif chars[i].isnumeric() and not chars[i+1].isnumeric():
                chars.insert(i+1, "_")
                i += 1
            else:
                i += 1
            # I want have for example 18_d|5_h|38_m|36_s

        formatted_string = [tuple(group.split("_")) for group in "".join(chars).split("|")]
        # recreates the string, then splits it first to have the number_keyword and then create the tuples

    return formatted_string


def parse_date(column: list) -> DataFrame:
    """Parses strings of column into datetime objects and returns a DataFrame

    :param column: list of str, strings to parse into date
    :return:
    """
    results_df = DataFrame(columns=['date_day', 'date_month', 'date_year', 'date_hours', 'date_minutes', 'date_seconds',
                                    'date_special_occasion'])
    country = ['AO', 'AR', 'AW', 'AU', 'AT', 'BD', 'BY', 'BE', 'BR', 'BG', 'BI', 'CA', 'CL', 'CO', 'HR', 'CW', 'CZ',
               'DK', 'DJ', 'DO', 'EG', 'England', 'EE', 'ECB', 'FI', 'FR', 'GE', 'DE', 'GR', 'HN', 'HK', 'HU', 'IS',
               'IN', 'IE', 'IsleOfMan', 'IL', 'IT', 'JM', 'JP', 'KE', 'KR', 'LV', 'LT', 'LU', 'MW', 'MX', 'MA', 'MZ',
               'NL', 'NZ', 'NI', 'NG', 'NO', 'PY', 'PE', 'PL', 'PT', 'PTE', 'RO', 'RU', 'SA', 'Scotland', 'RS', 'SG',
               'SK', 'SI', 'ZA', 'ES', 'SE', 'CH', 'TR', 'UA', 'AE', 'GB', 'US', 'VN', 'Wales']

    for string in column:
        date = dateparser.parse(string)  # Returns a datetime type
        row = [date.day, date.month, date.year, date.hour, date.minute, date.second]

        i = 0
        special_occasion = False
        while (not special_occasion) and i < len(country):
            holiday = holidays.CountryHoliday(country[i])
            if date.strftime("%m-%d-%Y") in holiday:
                special_occasion = True
            i += 1
        row.append(special_occasion)

        results_df.loc[0 if isnull(results_df.index.max()) else results_df.index.max()+1] = row

    return results_df


def parse_duration(column: list) -> DataFrame:
    """Parses strings of column into datetime objects and returns a DataFrame

    I assume that I receive the duration in one of the following format
    - number (milliseconds)

    :param column:
    :return:
    """
    results_df = DataFrame(columns=['date_days', 'date_hours', 'date_minutes', 'date_seconds'])

    for string in column:
        tuples = format_string_duration_parse(string)
        row = [0, 0, 0, 0]

        if tuples[0][1] == duration_allowed_word[8]:  # tuples == 'ms'
            seconds = int(int(tuples[0][0])/1000)
            row[3] = seconds % 60  # not right, only for test
            minutes = int(seconds / 60)
            row[2] = minutes % 60  # not right, only for test
            hours = int(minutes / 60)
            row[2] = hours % 24  # not right, only for test
            days = int(hours / 24)
            row[1] = days
        else:
            for group in tuples:
                if group[1] == duration_allowed_word[0] or group[1] == duration_allowed_word[1]:
                    row[0] += int(group[0])
                elif group[1] == duration_allowed_word[2] or group[1] == duration_allowed_word[3]:
                    row[1] += int(group[0])
                elif group[1] == duration_allowed_word[4] or group[1] == duration_allowed_word[5]:
                    row[2] += int(group[0])
                elif group[1] == duration_allowed_word[6] or group[1] == duration_allowed_word[7]:
                    row[3] += int(group[0])

        results_df.loc[0 if isnull(results_df.index.max()) else results_df.index.max()+1] = row  # append row

    return results_df


main()
