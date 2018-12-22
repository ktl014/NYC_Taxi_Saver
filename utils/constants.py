"""Constants"""
TRAIN = 'train'
TEST = 'test'

class Col():
    KEY = 'key'
    FA = 'fare_amount'
    LOG_FA = 'log_fare_amount'
    BIN_FA = 'fare_bin'

    PU_LONG = 'pickup_longitude'
    PU_LAT = 'pickup_latitude'
    DO_LONG = 'dropoff_longitude'
    DO_LAT = 'dropoff_latitude'
    COUNT = 'passenger_count'

    ABS_LAT_DIFF = 'abs_lat_diff'
    ABS_LON_DIFF = 'abs_lon_diff'
    MANHAT = 'manhattan'
    EUCLID = 'euclid'

    TRIP_DIST = 'trip_distance_km'
    LOG_TRIP = 'log_trip_distance'

    DATETIME = 'pickup_datetime'
    pickup_year = "pickup_Year"
    pickup_month = "pickup_Month"
    pickup_week = "pickup_Week"
    pickup_day = "pickup_Day"
    pickup_dayofweek = "pickup_Dayofweek"
    pickup_dayofyear = "pickup_Dayofyear"
    pickup_days_in_month = "pickup_Days_in_month"
    pickup_is_leap_year = "pickup_is_leap_year"
    pickup_is_month_end = "pickup_Is_month_end"
    pickup_is_month_start = "pickup_Is_month_start"
    pickup_is_quarter_end = "pickup_Is_quarter_end"
    pickup_is_quarter_start = "pickup_Is_quarter_start"
    pickup_is_year_end = "pickup_Is_year_end"
    pickup_is_year_start = "pickup_Is_year_start"
    pickup_hour = "pickup_Hour"
    pickup_minute = "pickup_Minute"
    pickup_second = "pickup_Second"
    pickup_days_in_year = "pickup_Days_in_year"
    pickup_frac_day = "pickup_frac_day"
    pickup_frac_week = "pickup_frac_week"
    pickup_frac_month = "pickup_frac_month"
    pickup_frac_year = "pickup_frac_year"
    pickup_elapsed = "pickup_Elapsed"

COORD = [Col.PU_LONG, Col.PU_LAT, Col.DO_LONG, Col.DO_LAT]
