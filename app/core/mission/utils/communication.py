from datetime import datetime
from typing import TypedDict

import numpy as np

from app.core.mission.utils.propagator import PropagationResults
from app.core.mission.utils.visibility import compute_visibility


class GroundStationConfig(TypedDict):
    station: str
    lat: float
    lon: float
    alt: float
    min_elevation_deg: float


class Window(TypedDict):
    start: datetime
    end: datetime
    max_ele: float


class StationResult(TypedDict):
    station: str
    elevation_deg: list
    visible: list
    windows: list[Window]


class Pass(TypedDict):
    station: str
    start: datetime
    end: datetime
    duration: float


class CommunicationResults(TypedDict):
    total_contact_time: float
    passes: list[Pass]
    passes_per_day: float


def compute_visibility_for_station(
    propagation_results: PropagationResults,
    ground_station: GroundStationConfig,
) -> StationResult:
    """
    Compute visibility analysis for a single ground station.

    This function calculates when a spacecraft is visible from a specific ground
    station based on the minimum elevation angle constraint. It processes the
    entire propagation trajectory and identifies visibility windows.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing the spacecraft trajectory data with keys:
        - 'time': list of datetime objects for each propagation step
        - 'r_eci': list of position vectors in ECI coordinates (km)
        - Other propagation state information
    ground_station : GroundStationConfig
        Dictionary containing ground station configuration:
        - 'station': name/identifier of the ground station (str)
        - 'lat': latitude in degrees, range -90 to 90 (float)
        - 'lon': longitude in degrees, range -180 to 180 (float)
        - 'alt': altitude above sea level in meters (float)
        - 'min_elevation_deg': minimum elevation angle for visibility in degrees (float)

    Returns
    -------
    StationResult
        TypedDict containing:
        - 'station': ground station name (str)
        - 'elevation_deg': array of elevation angles in degrees at each time step (np.ndarray)
        - 'visible': boolean array indicating visibility at each time step (np.ndarray)
        - 'windows': list of visibility windows, each containing start time,
          end time, and maximum elevation during the window (list[Window])

    Notes
    -----
    The visibility computation accounts for:
    - Minimum elevation angle constraints
    - Earth's rotation
    - Geometric line-of-sight calculations

    The function internally calls compute_visibility which handles the detailed
    geometric calculations for spacecraft-to-ground-station visibility.
    """
    visibility_results = compute_visibility(
        propagation_results, ground_station, ground_station["min_elevation_deg"]
    )

    return {
        "station": ground_station["station"],
        "elevation_deg": visibility_results["elevation_deg"],
        "visible": visibility_results["visible"],
        "windows": visibility_results["windows"],
    }


def compute_contact_duration_from_windows(windows: list | np.ndarray):
    """
    Calculate total contact duration by merging overlapping visibility windows.

    This function takes visibility windows from potentially multiple ground stations
    and merges any overlapping time periods to avoid double-counting. The total
    contact time represents the cumulative time when at least one ground station
    has visibility.

    Parameters
    ----------
    windows : list or np.ndarray
        List of window dictionaries, where each window contains:
        - 'start': datetime object marking window start (datetime)
        - 'end': datetime object marking window end (datetime)
        Additional keys may be present but are not used in this function.
        Windows can be from different ground stations and may overlap.

    Returns
    -------
    float
        Total contact duration in seconds after merging overlapping windows.
        Returns 0 if the windows list is empty.

    Notes
    -----
    Algorithm:
    1. Sort windows by start time
    2. Merge overlapping or contiguous windows
    3. Sum the duration of all merged windows

    Overlapping windows are combined into a single window spanning from the
    earliest start to the latest end time within the overlapping group.

    Examples
    --------
    If two windows overlap:
    - Window 1: 10:00-10:30
    - Window 2: 10:20-10:50
    Merged: 10:00-10:50 (50 minutes total, not 60 minutes)
    """
    # windows must be list or array of dicts with 'start', 'end'

    if not windows:
        return 0

    windows = sorted(windows, key=lambda x: x["start"])
    merged_windows = [windows[0]]

    for window in windows[1:]:
        if merged_windows[-1]["end"] > window["start"]:
            merged_windows[-1]["end"] = max(merged_windows[-1]["end"], window["end"])
        else:
            merged_windows.append(window)

    contact_dur = 0

    for window in merged_windows:
        contact_dur += (window["end"] - window["start"]).total_seconds()

    return contact_dur


def compute_passes(windows: list | np.ndarray):
    """
    Extract individual satellite passes from visibility windows.

    This function processes visibility windows from multiple ground stations and
    identifies discrete satellite passes. It handles overlapping windows by
    splitting them into separate passes, ensuring that simultaneous visibility
    from multiple stations is properly represented as distinct pass segments.

    Parameters
    ----------
    windows : list or np.ndarray
        List of window dictionaries, where each window must contain:
        - 'start': datetime object marking window start (datetime)
        - 'end': datetime object marking window end (datetime)
        - 'station': ground station identifier (str)
        Windows may overlap when multiple stations have simultaneous visibility.

    Returns
    -------
    list[Pass]
        List of pass dictionaries, where each pass contains:
        - 'station': ground station name (str)
        - 'start': pass start time (datetime)
        - 'end': pass end time (datetime)
        - 'duration': pass duration in seconds (float)
        Returns empty list if no windows are provided.

    Notes
    -----
    Pass Extraction Logic:
    - Windows are sorted chronologically by start time
    - If a new window starts before the previous one ends, the overlap is handled:
      * If the new window ends before the previous: skip it (fully contained)
      * Otherwise: create a new pass starting when the previous ends
    - Non-overlapping windows create separate passes
    - Each pass is associated with a single ground station

    This approach ensures that:
    1. Time periods are not double-counted across stations
    2. Each pass segment is attributed to the correct station
    3. Continuous coverage from overlapping stations is properly segmented

    Examples
    --------
    Station A visible: 10:00-10:30
    Station B visible: 10:20-10:50

    Results in two passes:
    - Pass 1: Station A, 10:00-10:30 (30 minutes)
    - Pass 2: Station B, 10:30-10:50 (20 minutes)
    """
    # windows must be list or array of dicts with 'start', 'end', 'station'

    if not windows:
        return []

    windows = sorted(windows, key=lambda x: x["start"])
    merged_windows = [windows[0]]

    for window in windows[1:]:
        if merged_windows[-1]["end"] > window["start"]:
            if merged_windows[-1]["end"] > window["end"]:
                continue
            else:
                new_window = window.copy()
                new_window["start"] = merged_windows[-1]["end"]
                merged_windows.append(new_window)
        else:
            merged_windows.append(window)

    for idx in range(len(merged_windows)):
        merged_windows[idx]["duration"] = (
            merged_windows[idx]["end"] - merged_windows[idx]["start"]
        ).total_seconds()

    # List of dicts{station, start, end, duration}. Each dict = 1 pass
    return merged_windows


def aggregate_ground_stations(
    station_results: list[StationResult],
) -> CommunicationResults:
    """
    Aggregate visibility results from multiple ground stations.

    This function combines visibility data from all ground stations to produce
    comprehensive communication statistics for the entire ground station network.
    It calculates total contact time (accounting for overlaps), extracts individual
    passes, and computes the average number of passes per day.

    Parameters
    ----------
    station_results : list[StationResult]
        List of visibility results for each ground station, where each result contains:
        - 'station': ground station name (str)
        - 'elevation_deg': elevation angles array (np.ndarray)
        - 'visible': visibility boolean array (np.ndarray)
        - 'windows': list of visibility windows with 'start', 'end', 'max_ele' (list[Window])

    Returns
    -------
    CommunicationResults
        TypedDict containing aggregated communication metrics:
        - 'total_contact_time': total time (in seconds) when at least one station
          has visibility, with overlapping windows properly merged (float)
        - 'passes': list of all satellite passes across all stations, chronologically
          ordered with non-overlapping time segments (list[Pass])
        - 'passes_per_day': average number of passes per day over the mission
          duration, calculated as total passes divided by mission duration in days (float)

    Notes
    -----
    The aggregation process:
    1. Collects all visibility windows from all stations
    2. Associates each window with its corresponding station
    3. Computes total contact time by merging overlapping windows
    4. Extracts discrete passes, handling overlaps between stations
    5. Calculates pass frequency normalized to per-day basis

    Mission duration is determined from the earliest window start to the latest
    window end across all stations. If the duration is zero or no windows exist,
    passes_per_day returns 0.0.

    This aggregation is useful for:
    - Mission planning: understanding overall ground contact availability
    - Data budget analysis: estimating data downlink opportunities
    - Network optimization: identifying coverage gaps or redundancies
    """
    all_station_windows = []

    for station in station_results:
        updated_windows = []
        for window in station["windows"]:
            window = {**window, "station": station["station"]}
            updated_windows.append(window)

        all_station_windows += updated_windows

    total_contact_time = compute_contact_duration_from_windows(all_station_windows)
    passes = compute_passes(all_station_windows)

    mission_start = min(w["start"] for w in all_station_windows)
    mission_end = max(w["end"] for w in all_station_windows)
    days = (mission_end - mission_start).total_seconds() / 86400

    return {
        "total_contact_time": total_contact_time,
        "passes": passes,
        "passes_per_day": len(passes) / days if days > 0 else 0.0,
    }


def compute_communication(
    propagation_results: PropagationResults,
    ground_stations: list[GroundStationConfig],
) -> CommunicationResults:
    """
    Compute comprehensive communication analysis for a satellite mission.

    This is the main entry point for communication analysis. It processes the
    spacecraft trajectory against multiple ground stations to determine visibility
    windows, contact opportunities, and overall communication statistics for the
    ground station network.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing the complete spacecraft trajectory data:
        - 'time': array of datetime objects for each propagation time step (list[datetime])
        - 'r_eci': array of position vectors in ECI coordinates (km) (list[np.ndarray])
        - 'v_eci': array of velocity vectors in ECI coordinates (km/s) (list[np.ndarray])
        The trajectory should span the desired analysis period.
    ground_stations : list[GroundStationConfig]
        List of ground station configurations, where each station is a dictionary:
        - 'station': unique identifier/name of the station (str)
        - 'lat': latitude in degrees, range -90 to 90 (float)
        - 'lon': longitude in degrees, range -180 to 180 (float)
        - 'alt': altitude above sea level in meters (float)
        - 'min_elevation_deg': minimum elevation angle for communication in degrees,
          typically 5-10 degrees to account for atmospheric effects and horizon
          obstructions (float)

    Returns
    -------
    CommunicationResults
        TypedDict containing comprehensive communication metrics:
        - 'total_contact_time': cumulative time (seconds) when at least one ground
          station has line-of-sight to the spacecraft, with overlaps merged (float)
        - 'passes': chronologically ordered list of all satellite passes over all
          stations, where each pass includes station name, start/end times, and
          duration (list[Pass])
        - 'passes_per_day': average frequency of passes per day, useful for planning
          communication schedules and data downlink budgets (float)

    Notes
    -----
    The analysis workflow:
    1. For each ground station:
       - Compute elevation angles throughout the trajectory
       - Identify visibility windows based on minimum elevation constraint
       - Extract window timing and maximum elevation during each pass
    2. Aggregate results across all stations:
       - Merge overlapping windows to compute total contact time
       - Extract non-overlapping passes for scheduling purposes
       - Calculate pass statistics (frequency, distribution)

    Key considerations:
    - Minimum elevation angle accounts for atmospheric effects, antenna patterns,
      and line-of-sight obstructions near the horizon
    - Overlapping visibility from multiple stations is handled to avoid
      double-counting contact time
    - Pass extraction creates non-overlapping time segments attributed to specific
      stations for downlink scheduling

    Typical minimum elevation angles:
    - 5° for most satellite communications
    - 10° for high-precision applications or obstructed sites
    - 0° for theoretical horizon visibility (rarely used)

    Applications:
    - Mission planning: assess communication coverage sufficiency
    - Data budget: estimate downlink opportunities and data volumes
    - Station network design: optimize number and placement of ground stations
    - Operations scheduling: plan communication windows and command uploads
    """
    station_results = []

    for ground_station in ground_stations:
        station_results.append(compute_visibility_for_station(propagation_results, ground_station))

    return aggregate_ground_stations(station_results)
