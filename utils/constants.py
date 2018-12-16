"""Constants"""
TRAIN = 'train'
TEST = 'test'

class Col():
    KEY = 'key'
    FA = 'fare_amount'
    LOG_FA = 'log_fare_amount'

    PU_LONG = 'pickup_longitude'
    PU_LAT = 'pickup_latitude'
    DO_LONG = 'dropoff_longitude'
    DO_LAT = 'dropoff_latitude'
    COUNT = 'passenger_count'

    ABS_LAT_DIFF = 'abs_lat_diff'
    ABS_LON_DIFF = 'abs_lon_diff'

    TRIP_DIST = 'trip_distance_km'
    LOG_TRIP = 'log_trip_distance'

    DATETIME = 'pickup_datetime'
    YEAR = 'year'
    MONTH = 'month'
    MONTH_NAME = 'month_name'
    MONTH_YEAR = 'month_year'
    WEEK_DAY = 'week_day'
    DAY = 'day'
    HOUR = 'hour'


COORD = [Col.PU_LONG, Col.PU_LAT, Col.DO_LONG, Col.DO_LAT]