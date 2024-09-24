import math

def CalUTtoJD(Y, M, D, UT):
    """
    Converts year, month, day, and UT time to Julian date.

    Parameters:
        Y (int): year
        M (int): month
        D (int): day
        UT (float): UT time in hours

    Returns:
        float: Julian date
    """
    # Adjust year and month if month is less than or equal to 2
    if M <= 2:
        y = Y - 1
        m = M + 12
    elif M > 2:
        y = Y
        m = M
    else:
        raise ValueError("Incorrect format for M")

    # Determine if B should be computed based on year and month
    if Y > 1582 or (Y == 1582 and M > 10) or (Y == 1582 and M == 10 and D >= 15):
        B = math.floor(y / 400) - math.floor(y / 100)
    else:
        B = -2

    # Calculate the Julian date
    JD = (math.floor(365.25 * y) +
          math.floor(30.6001 * (m + 1)) +
          B + 1720996.5 + D + UT / 24)

    return JD


def dayOfYearToMonthDay(year, dayOfYear):
    """
    Converts the day of the year to the corresponding month and day for a given year.

    Parameters:
        year (int): Year in question (e.g., 2024)
        dayOfYear (float): Day of the year (1 through 365 or 366)

    Returns:
        month (int): Month (1 through 12)
        day (int): Day of the month (1 through 31)
        frac (float): Fraction of the day (if any)
    """
    # Check if the year is a leap year
    if isLeapYear(year):
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Leap year
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Regular year

    # Initialize month
    month = 1

    # Subtract days in each month until finding the correct month
    while dayOfYear > days_in_month[month - 1]:
        dayOfYear -= days_in_month[month - 1]
        month += 1

    # The remaining dayOfYear is the day of the month
    day = int(dayOfYear)
    frac = dayOfYear % 1

    return month, day, frac


def isLeapYear(year):
    """
    Determines if a given year is a leap year.

    Parameters:
        year (int): Year in question (e.g., 2024)

    Returns:
        leap (bool): True if leap year, False otherwise
    """
    # Check if the given year is a leap year
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return True
    else:
        return False


def fracDaytoHMS(frac):
    """
    Converts a fraction of a day into hours, minutes, and seconds.

    Parameters:
        frac (float): Fraction of a day (e.g., 0.5 for noon)

    Returns:
        hours (int): Hour of the day (0 to 23)
        minutes (int): Minute of the hour (0 to 59)
        seconds (float): Second of the minute (0 to 59)
    """
    # Total seconds in a day
    total_seconds_in_day = 24 * 3600

    # Convert fraction of day to total seconds
    total_seconds = frac * total_seconds_in_day

    # Get hours
    hours = int(total_seconds // 3600)

    # Get remaining seconds after hours
    remaining_seconds = total_seconds - hours * 3600

    # Get minutes
    minutes = int(remaining_seconds // 60)

    # Get seconds
    seconds = remaining_seconds - minutes * 60

    return hours, minutes, seconds


def timeconverter(T, input_scale, output_scale):
    """
    Converts time between different time scales, including UTC, UT1, TT, and TAI.
    It takes an input time and converts it to the specified output time format.

    Parameters:
        T (float): Time in the input time scale (in seconds)
        input_scale (str): The time scale of the input time ("UTC", "UT1", "TT", or "TAI")
        output_scale (str): The desired time scale for the output ("UTC", "UT1", "TT", or "TAI")

    Returns:
        float: Time converted to the specified output time scale
    """

    # Convert inputs to uppercase to make case-insensitive
    input_scale = input_scale.upper()
    output_scale = output_scale.upper()

    # Identify output time scale
    if output_scale in ["UTC", "UNIVERSAL COORDINATED TIME"]:
        ind = 1
    elif output_scale in ["UT", "UT1"]:
        ind = 2
    elif output_scale in ["TT", "TERRESTRIAL TIME"]:
        ind = 3
    elif output_scale in ["TAI", "INTERNATIONAL ATOMIC TIME"]:
        ind = 4
    else:
        raise ValueError("Invalid output time scale")

    # Convert input time to Terrestrial Time (TT)
    if input_scale in ["UTC", "UNIVERSAL COORDINATED TIME"]:
        T += 64.184  # Convert from UTC to TT
    elif input_scale in ["UT", "UT1"]:
        T -= 0.649232  # Convert from UT1 to UTC
        T += 64.184  # Convert from UTC to TT
    elif input_scale in ["TAI", "INTERNATIONAL ATOMIC TIME"]:
        T -= 32.0  # Convert from TAI to UTC
        T += 64.184  # Convert from UTC to TT
    elif input_scale in ["TT", "TERRESTRIAL TIME"]:
        pass  # No conversion needed, already in TT
    else:
        raise ValueError("Invalid input time scale")

    # Convert from TT to the desired output time scale
    if ind == 1:
        T -= 64.184  # Convert from TT to UTC
    elif ind == 2:
        T -= 64.184  # Convert from TT to UTC
        T += 0.649232  # Convert from UTC to UT1
    elif ind == 3:
        pass  # No conversion needed, already in TT
    elif ind == 4:
        T -= 64.184  # Convert from TT to UTC
        T += 32.0  # Convert from UTC to TAI

    return T