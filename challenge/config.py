from enum import Enum

TOP_10_FEATURES = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

MORNING_MIN: str = "05:00"
MORNING_MAX: str = "11:59"
AFTERNOON_MIN: str = "12:00"
AFTERNOON_MAX: str = "18:59"
EVENING_MAX: str = "19:00"
EVENING_MIN: str = "23:59"
NIGHT_MIN: str = "00:00"
NIGHT_MAX: str = "4:59"

class PeriodDay(Enum):
    NIGHT = "noche"
    AFTERNOON = "tarde"
    MORNING = "mañana"


HIGH_SEASON_RANGE1_MIN: str = '15-Dec'
HIGH_SEASON_RANGE1_MAX: str = '31-Dec'
HIGH_SEASON_RANGE2_MIN: str = '1-Jan'
HIGH_SEASON_RANGE2_MAX: str = '3-Mar'
HIGH_SEASON_RANGE3_MIN: str = '15-Jul'
HIGH_SEASON_RANGE3_MAX: str = '31-Jul'
HIGH_SEASON_RANGE4_MIN: str = '11-Sep'
HIGH_SEASON_RANGE4_MAX: str = '30-Sep'

DELAY_THRESHOLD_IN_MINUTES: int = 15

THRESHOLD_PREDICT: float = 0.5