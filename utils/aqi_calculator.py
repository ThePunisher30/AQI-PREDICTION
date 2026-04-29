"""AQI Calculator — EPA breakpoint interpolation for PM2.5 and PM10."""

import numpy as np

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0.0, 54.999999, 0, 50),
    (55.0, 154.999999, 51, 100),
    (155.0, 254.999999, 101, 150),
    (255.0, 354.999999, 151, 200),
    (355.0, 424.999999, 201, 300),
    (425.0, 504.999999, 301, 400),
    (505.0, 604.999999, 401, 500),
]

HEALTH_CATEGORIES = [
    (0, 50, "Good", "green"),
    (51, 100, "Moderate", "yellow"),
    (101, 150, "Unhealthy for Sensitive Groups", "orange"),
    (151, 200, "Unhealthy", "red"),
    (201, 300, "Very Unhealthy", "purple"),
    (301, 500, "Hazardous", "maroon"),
]


def aqi_from_concentration(c, breakpoints):
    """Compute AQI from a single pollutant concentration using EPA breakpoint interpolation."""
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return np.nan
    c = max(float(c), 0.0)
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
    return 500.0


def compute_aqi(pm25, pm10):
    """Compute overall AQI as max of PM2.5 AQI and PM10 AQI."""
    aqi_pm25 = aqi_from_concentration(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = aqi_from_concentration(pm10, PM10_BREAKPOINTS)
    return max(aqi_pm25, aqi_pm10)


def get_health_category(aqi):
    """Map AQI value to health category and color."""
    for low, high, category, color in HEALTH_CATEGORIES:
        if low <= aqi <= high:
            return category, color
    return "Hazardous", "maroon"


if __name__ == "__main__":
    # Quick test
    print(f"PM2.5=35, PM10=80 → AQI={compute_aqi(35, 80):.1f}")
    print(f"Category: {get_health_category(compute_aqi(35, 80))}")
