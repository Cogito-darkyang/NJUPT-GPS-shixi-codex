from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0
OMEGA_E = 7.2921151467e-5
MU_GPS = 3.986005e14
MU_BDS = 3.986004418e14
F_REL = -4.442807633e-10
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
BDT_MINUS_GPS = -14.0

FREQ_GPS_L1 = 1_575_420_000.0
FREQ_BDS_B1I = 1_561_098_000.0
LAMBDA_GPS_L1 = C_LIGHT / FREQ_GPS_L1
LAMBDA_BDS_B1I = C_LIGHT / FREQ_BDS_B1I

IONO_ALPHA_ZERO = (0.0, 0.0, 0.0, 0.0)
IONO_BETA_ZERO = (0.0, 0.0, 0.0, 0.0)


@dataclass
class NavRecord:
    sv: str
    system: str
    toc: datetime
    af0: float
    af1: float
    af2: float
    iode: float
    crs: float
    dn: float
    m0: float
    cuc: float
    ecc: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    codes: float
    week: int
    l2p: float
    svacc: float
    svhealth: float
    tgd: float
    iodc: float
    trans: float
    fit: float


@dataclass
class Observation:
    sv: str
    system: str
    pseudorange: Optional[float]
    carrier: Optional[float]
    snr: Optional[float]
    smoothed_pseudorange: Optional[float] = None


@dataclass
class Epoch:
    dt: datetime
    gps_sow: float
    seconds_from_start: float
    observations: Dict[str, Observation]


@dataclass
class ObsHeader:
    obs_types: Dict[str, List[str]]
    approx_xyz: np.ndarray
    interval: float
    first_time: Optional[datetime]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPS practice experiment figures for sections 2.2-2.4."
    )
    parser.add_argument(
        "--rinex-dir",
        type=Path,
        default=None,
        help="Directory containing RINEX files. Defaults to auto-detected point 1 data.",
    )
    parser.add_argument("--prefix", default="2160B3", help="RINEX file prefix.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output directory."
    )
    parser.add_argument(
        "--hatch-window",
        type=int,
        default=120,
        help="Maximum Hatch smoothing window in epochs.",
    )
    parser.add_argument(
        "--elevation-mask",
        type=float,
        default=10.0,
        help="Elevation mask in degrees for positioning.",
    )
    return parser.parse_args()


def discover_rinex_dir(base: Path) -> Path:
    candidates: List[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        rinex = child / "20250609" / "rinex"
        if rinex.exists():
            candidates.append(rinex)
    if not candidates:
        for rinex in base.rglob("rinex"):
            if rinex.is_dir():
                candidates.append(rinex)
    for rinex in candidates:
        if (rinex / "2160B3.25O").exists():
            return rinex
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not auto-detect the RINEX directory.")


def rinex_float_values(text: str) -> List[float]:
    pattern = r"[+-]?\d+\.\d+(?:[DE][+-]?\d+)?|[+-]?\d+(?:[DE][+-]?\d+)"
    return [float(match.replace("D", "E")) for match in re.findall(pattern, text)]


def gps_week_sow(dt: datetime) -> Tuple[int, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    total_seconds = (dt - GPS_EPOCH).total_seconds()
    return int(total_seconds // 604800.0), total_seconds % 604800.0


def normalize_gnss_seconds(seconds: float) -> float:
    if seconds > 302400.0:
        seconds -= 604800.0
    elif seconds < -302400.0:
        seconds += 604800.0
    return seconds


def parse_nav_file(path: Path, system: str) -> Dict[str, List[NavRecord]]:
    records: Dict[str, List[NavRecord]] = {}
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if "END OF HEADER" in line:
                break
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue

            sv = line[:3].strip()
            continuation = [handle.readline() for _ in range(7)]
            if not sv.startswith(system):
                continue

            parts = line[:23].split()
            if len(parts) < 7:
                continue
            year, month, day, hour, minute, second = map(int, parts[1:7])
            toc = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            first_values = rinex_float_values(line[23:])
            values: List[float] = []
            for row in continuation:
                values.extend(rinex_float_values(row))
            if len(first_values) < 3 or len(values) < 28:
                continue

            record = NavRecord(
                sv=sv,
                system=system,
                toc=toc,
                af0=first_values[0],
                af1=first_values[1],
                af2=first_values[2],
                iode=values[0],
                crs=values[1],
                dn=values[2],
                m0=values[3],
                cuc=values[4],
                ecc=values[5],
                cus=values[6],
                sqrt_a=values[7],
                toe=values[8],
                cic=values[9],
                omega0=values[10],
                cis=values[11],
                i0=values[12],
                crc=values[13],
                omega=values[14],
                omega_dot=values[15],
                idot=values[16],
                codes=values[17],
                week=int(round(values[18])),
                l2p=values[19],
                svacc=values[20],
                svhealth=values[21],
                tgd=values[22],
                iodc=values[23],
                trans=values[24],
                fit=values[25],
            )
            records.setdefault(sv, []).append(record)

    for sv_records in records.values():
        sv_records.sort(key=lambda rec: rec.toe)
    return records


def parse_observation_header(path: Path) -> ObsHeader:
    obs_types: Dict[str, List[str]] = {}
    approx_xyz: Optional[np.ndarray] = None
    interval = 1.0
    first_time: Optional[datetime] = None

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if "APPROX POSITION XYZ" in line:
                approx_xyz = np.array(
                    [
                        float(line[0:14]),
                        float(line[14:28]),
                        float(line[28:42]),
                    ],
                    dtype=float,
                )
            elif "SYS / # / OBS TYPES" in line:
                system = line[0]
                count = int(line[3:6])
                types = line[7:60].split()
                while len(types) < count:
                    continuation = next(handle)
                    types.extend(continuation[7:60].split())
                obs_types[system] = types[:count]
            elif "INTERVAL" in line:
                interval = float(line[:10])
            elif "TIME OF FIRST OBS" in line:
                fields = line[:43].split()
                if len(fields) >= 6:
                    y, m, d, hh, mm = map(int, fields[:5])
                    ss = float(fields[5])
                    first_time = datetime(
                        y,
                        m,
                        d,
                        hh,
                        mm,
                        int(ss),
                        int(round((ss - int(ss)) * 1_000_000)),
                        tzinfo=timezone.utc,
                    )
            elif "END OF HEADER" in line:
                break

    if approx_xyz is None:
        raise ValueError(f"Missing APPROX POSITION XYZ in {path}")
    return ObsHeader(obs_types=obs_types, approx_xyz=approx_xyz, interval=interval, first_time=first_time)


def field_float(text: str) -> Optional[float]:
    token = text[:14].strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_observation_file(path: Path) -> Tuple[ObsHeader, List[Epoch]]:
    header = parse_observation_header(path)
    index_by_system = {
        system: {obs_type: idx for idx, obs_type in enumerate(types)}
        for system, types in header.obs_types.items()
    }
    first_dt: Optional[datetime] = None
    epochs: List[Epoch] = []

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if "END OF HEADER" in line:
                break

        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            year, month, day, hour, minute = map(int, parts[1:6])
            seconds = float(parts[6])
            flag = int(parts[7])
            nsat = int(parts[8])
            sat_lines = [handle.readline().rstrip("\n") for _ in range(nsat)]
            if flag > 1:
                continue

            dt = datetime(
                year,
                month,
                day,
                hour,
                minute,
                int(seconds),
                int(round((seconds - int(seconds)) * 1_000_000)),
                tzinfo=timezone.utc,
            )
            if first_dt is None:
                first_dt = dt
            _, sow = gps_week_sow(dt)
            observations: Dict[str, Observation] = {}

            for sat_line in sat_lines:
                sv = sat_line[:3].strip()
                if len(sv) < 2:
                    continue
                system = sv[0]
                if system not in ("G", "C"):
                    continue
                type_index = index_by_system.get(system)
                if type_index is None:
                    continue

                code_type = "C1C" if system == "G" else "C1I"
                phase_type = "L1C" if system == "G" else "L1I"
                snr_type = "S1C" if system == "G" else "S1I"
                needed = 3 + 16 * len(header.obs_types[system])
                sat_line = sat_line.ljust(needed)

                def read_obs(obs_type: str) -> Optional[float]:
                    idx = type_index.get(obs_type)
                    if idx is None:
                        return None
                    return field_float(sat_line[3 + 16 * idx : 3 + 16 * (idx + 1)])

                observations[sv] = Observation(
                    sv=sv,
                    system=system,
                    pseudorange=read_obs(code_type),
                    carrier=read_obs(phase_type),
                    snr=read_obs(snr_type),
                )

            epochs.append(
                Epoch(
                    dt=dt,
                    gps_sow=sow,
                    seconds_from_start=(dt - first_dt).total_seconds(),
                    observations=observations,
                )
            )

    return header, epochs


def ecef_to_llh(xyz: np.ndarray) -> Tuple[float, float, float]:
    a = 6378137.0
    e2 = 6.69437999014e-3
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - e2 * n / (n + h)))
    return lat, lon, h


def ecef_to_enu_matrix(ref_xyz: np.ndarray) -> np.ndarray:
    lat, lon, _ = ecef_to_llh(ref_xyz)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def az_el(receiver_xyz: np.ndarray, satellite_xyz: np.ndarray) -> Tuple[float, float]:
    enu = ecef_to_enu_matrix(receiver_xyz) @ (satellite_xyz - receiver_xyz)
    east, north, up = enu
    horiz = math.hypot(east, north)
    azimuth = math.atan2(east, north) % (2.0 * math.pi)
    elevation = math.atan2(up, horiz)
    return azimuth, elevation


def solve_kepler(m: float, ecc: float) -> float:
    eccentric_anomaly = m
    for _ in range(14):
        delta = (eccentric_anomaly - ecc * math.sin(eccentric_anomaly) - m) / (
            1.0 - ecc * math.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= delta
        if abs(delta) < 1e-13:
            break
    return eccentric_anomaly


def select_ephemeris(nav: Dict[str, List[NavRecord]], sv: str, transmit_sow: float) -> Optional[NavRecord]:
    records = nav.get(sv)
    if not records:
        return None
    return min(records, key=lambda rec: abs(normalize_gnss_seconds(transmit_sow - rec.toe)))


def satellite_position_clock(record: NavRecord, transmit_sow: float) -> Tuple[np.ndarray, float]:
    mu = MU_GPS if record.system == "G" else MU_BDS
    a = record.sqrt_a * record.sqrt_a
    tk = normalize_gnss_seconds(transmit_sow - record.toe)
    mean_motion = math.sqrt(mu / (a * a * a)) + record.dn
    mean_anomaly = record.m0 + mean_motion * tk
    eccentric_anomaly = solve_kepler(mean_anomaly, record.ecc)
    sin_e = math.sin(eccentric_anomaly)
    cos_e = math.cos(eccentric_anomaly)
    true_anomaly = math.atan2(math.sqrt(1.0 - record.ecc * record.ecc) * sin_e, cos_e - record.ecc)
    argument_latitude = true_anomaly + record.omega

    du = record.cus * math.sin(2.0 * argument_latitude) + record.cuc * math.cos(2.0 * argument_latitude)
    dr = record.crs * math.sin(2.0 * argument_latitude) + record.crc * math.cos(2.0 * argument_latitude)
    di = record.cis * math.sin(2.0 * argument_latitude) + record.cic * math.cos(2.0 * argument_latitude)

    u = argument_latitude + du
    r = a * (1.0 - record.ecc * cos_e) + dr
    inc = record.i0 + record.idot * tk + di
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    omega = record.omega0 + (record.omega_dot - OMEGA_E) * tk - OMEGA_E * record.toe
    x = x_orb * math.cos(omega) - y_orb * math.cos(inc) * math.sin(omega)
    y = x_orb * math.sin(omega) + y_orb * math.cos(inc) * math.cos(omega)
    z = y_orb * math.sin(inc)

    _, toc_sow_gps = gps_week_sow(record.toc)
    toc_sow = toc_sow_gps if record.system == "G" else toc_sow_gps + BDT_MINUS_GPS
    tc = normalize_gnss_seconds(transmit_sow - toc_sow)
    clock = (
        record.af0
        + record.af1 * tc
        + record.af2 * tc * tc
        + F_REL * record.ecc * record.sqrt_a * sin_e
        - record.tgd
    )
    return np.array([x, y, z], dtype=float), clock


def klobuchar_delay_m(
    receiver_xyz: np.ndarray,
    azimuth: float,
    elevation: float,
    gps_sow: float,
    alpha: Sequence[float] = IONO_ALPHA_ZERO,
    beta: Sequence[float] = IONO_BETA_ZERO,
) -> float:
    if elevation <= 0.0:
        return 0.0
    lat, lon, _ = ecef_to_llh(receiver_xyz)
    elevation_sc = elevation / math.pi
    lat_u = lat / math.pi
    lon_u = lon / math.pi

    psi = 0.0137 / (elevation_sc + 0.11) - 0.022
    lat_i = lat_u + psi * math.cos(azimuth)
    lat_i = min(0.416, max(-0.416, lat_i))
    lon_i = lon_u + psi * math.sin(azimuth) / max(math.cos(lat_i * math.pi), 1e-12)
    lat_m = lat_i + 0.064 * math.cos((lon_i - 1.617) * math.pi)
    local_time = (43200.0 * lon_i + gps_sow) % 86400.0

    amp = sum(alpha[i] * lat_m**i for i in range(4))
    per = sum(beta[i] * lat_m**i for i in range(4))
    amp = max(0.0, amp)
    per = max(72000.0, per)
    x = 2.0 * math.pi * (local_time - 50400.0) / per
    f = 1.0 + 16.0 * (0.53 - elevation_sc) ** 3
    if abs(x) < 1.57:
        delay_seconds = f * (5e-9 + amp * (1.0 - x * x / 2.0 + x**4 / 24.0))
    else:
        delay_seconds = f * 5e-9
    return delay_seconds * C_LIGHT


def troposphere_delay_m(receiver_xyz: np.ndarray, elevation: float) -> float:
    if elevation <= math.radians(3.0):
        return 0.0
    lat, _, height = ecef_to_llh(receiver_xyz)
    height = max(-100.0, min(5000.0, height))
    pressure = 1013.25 * (1.0 - 2.2557e-5 * height) ** 5.2568
    temperature = 291.15 - 0.0065 * height
    water_vapor_pressure = 11.0
    zhd = 0.0022768 * pressure / (1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * height / 1000.0)
    zwd = 0.002277 * (1255.0 / temperature + 0.05) * water_vapor_pressure
    mapping = 1.0 / max(math.sin(elevation), 0.08)
    return (zhd + zwd) * mapping


def rotated_for_earth_spin(satellite_xyz: np.ndarray, geometric_range: float) -> np.ndarray:
    theta = OMEGA_E * geometric_range / C_LIGHT
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [
            cos_t * satellite_xyz[0] + sin_t * satellite_xyz[1],
            -sin_t * satellite_xyz[0] + cos_t * satellite_xyz[1],
            satellite_xyz[2],
        ],
        dtype=float,
    )


def observation_wavelength(system: str) -> float:
    return LAMBDA_GPS_L1 if system == "G" else LAMBDA_BDS_B1I


def solve_epoch_position(
    epoch: Epoch,
    nav_gps: Dict[str, List[NavRecord]],
    nav_bds: Dict[str, List[NavRecord]],
    reference_xyz: np.ndarray,
    use_smoothed: bool,
    elevation_mask_rad: float,
) -> Tuple[Optional[np.ndarray], List[str], Optional[np.ndarray]]:
    x = np.array([1.0, 1.0, 1.0], dtype=float)
    gps_clock_m = 0.0
    bds_clock_m = 0.0
    final_residuals: Optional[np.ndarray] = None
    final_used: List[str] = []

    for _ in range(10):
        rows: List[List[float]] = []
        residuals: List[float] = []
        weights: List[float] = []
        used: List[str] = []
        angle_xyz = x if np.linalg.norm(x) > 6.0e6 else reference_xyz

        for sv, obs in epoch.observations.items():
            if obs.system not in ("G", "C"):
                continue
            if obs.system == "C" and int(sv[1:]) <= 5:
                continue
            pseudorange = obs.smoothed_pseudorange if use_smoothed and obs.smoothed_pseudorange else obs.pseudorange
            if pseudorange is None or pseudorange <= 0.0:
                continue

            if obs.system == "G":
                transmit_sow = epoch.gps_sow - pseudorange / C_LIGHT
                nav_record = select_ephemeris(nav_gps, sv, transmit_sow)
            else:
                transmit_sow = epoch.gps_sow + BDT_MINUS_GPS - pseudorange / C_LIGHT
                nav_record = select_ephemeris(nav_bds, sv, transmit_sow)
            if nav_record is None:
                continue

            sat_xyz, sat_clock_s = satellite_position_clock(nav_record, transmit_sow)
            approx_range = np.linalg.norm(sat_xyz - angle_xyz)
            sat_xyz = rotated_for_earth_spin(sat_xyz, approx_range)
            azimuth, elevation = az_el(angle_xyz, sat_xyz)
            if np.linalg.norm(x) > 6.0e6 and elevation < elevation_mask_rad:
                continue

            iono = klobuchar_delay_m(angle_xyz, azimuth, elevation, epoch.gps_sow)
            tropo = troposphere_delay_m(angle_xyz, elevation)
            corrected_p = pseudorange - iono - tropo

            geometric_range = np.linalg.norm(sat_xyz - x)
            if geometric_range <= 0.0:
                continue
            clock_m = gps_clock_m if obs.system == "G" else bds_clock_m
            predicted = geometric_range + clock_m - C_LIGHT * sat_clock_s
            residual = corrected_p - predicted
            line_of_sight = (x - sat_xyz) / geometric_range
            row = [
                line_of_sight[0],
                line_of_sight[1],
                line_of_sight[2],
                1.0 if obs.system == "G" else 0.0,
                1.0 if obs.system == "C" else 0.0,
            ]

            snr = obs.snr if obs.snr is not None and obs.snr > 0.0 else 40.0
            elevation_weight = max(math.sin(max(elevation, math.radians(5.0))), 0.12)
            snr_weight = min(max(snr, 20.0), 55.0) / 55.0
            rows.append(row)
            residuals.append(residual)
            weights.append(elevation_weight * math.sqrt(snr_weight))
            used.append(sv)

        if (
            len(rows) < 6
            or not any(sv.startswith("G") for sv in used)
            or not any(sv.startswith("C") for sv in used)
        ):
            return None, used, None

        h = np.asarray(rows, dtype=float)
        v = np.asarray(residuals, dtype=float)
        w = np.asarray(weights, dtype=float)
        hw = h * w[:, None]
        vw = v * w
        try:
            dx, *_ = np.linalg.lstsq(hw, vw, rcond=None)
        except np.linalg.LinAlgError:
            return None, used, None

        x += dx[:3]
        gps_clock_m += dx[3]
        bds_clock_m += dx[4]
        final_residuals = v
        final_used = used
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    if final_residuals is not None and len(final_residuals) >= 8:
        mask = np.abs(final_residuals - np.median(final_residuals)) < 40.0
        if mask.sum() >= 6 and mask.sum() < len(final_residuals):
            # One refinement pass after discarding very large code residuals.
            rows = []
            residuals = []
            weights = []
            used = []
            angle_xyz = x
            for sv, obs in epoch.observations.items():
                if obs.system == "C" and int(sv[1:]) <= 5:
                    continue
                pseudorange = obs.smoothed_pseudorange if use_smoothed and obs.smoothed_pseudorange else obs.pseudorange
                if pseudorange is None or pseudorange <= 0.0:
                    continue
                if obs.system == "G":
                    transmit_sow = epoch.gps_sow - pseudorange / C_LIGHT
                    nav_record = select_ephemeris(nav_gps, sv, transmit_sow)
                elif obs.system == "C":
                    transmit_sow = epoch.gps_sow + BDT_MINUS_GPS - pseudorange / C_LIGHT
                    nav_record = select_ephemeris(nav_bds, sv, transmit_sow)
                else:
                    continue
                if nav_record is None:
                    continue
                sat_xyz, sat_clock_s = satellite_position_clock(nav_record, transmit_sow)
                sat_xyz = rotated_for_earth_spin(sat_xyz, np.linalg.norm(sat_xyz - x))
                azimuth, elevation = az_el(angle_xyz, sat_xyz)
                if elevation < elevation_mask_rad:
                    continue
                iono = klobuchar_delay_m(angle_xyz, azimuth, elevation, epoch.gps_sow)
                tropo = troposphere_delay_m(angle_xyz, elevation)
                corrected_p = pseudorange - iono - tropo
                geometric_range = np.linalg.norm(sat_xyz - x)
                clock_m = gps_clock_m if obs.system == "G" else bds_clock_m
                residual = corrected_p - (geometric_range + clock_m - C_LIGHT * sat_clock_s)
                line_of_sight = (x - sat_xyz) / geometric_range
                rows.append(
                    [
                        line_of_sight[0],
                        line_of_sight[1],
                        line_of_sight[2],
                        1.0 if obs.system == "G" else 0.0,
                        1.0 if obs.system == "C" else 0.0,
                    ]
                )
                residuals.append(residual)
                snr = obs.snr if obs.snr is not None and obs.snr > 0.0 else 40.0
                weights.append(max(math.sin(elevation), 0.12) * math.sqrt(min(max(snr, 20.0), 55.0) / 55.0))
                used.append(sv)
            if (
                len(rows) >= 6
                and any(sv.startswith("G") for sv in used)
                and any(sv.startswith("C") for sv in used)
            ):
                h = np.asarray(rows, dtype=float)
                v = np.asarray(residuals, dtype=float)
                w = np.asarray(weights, dtype=float)
                try:
                    dx, *_ = np.linalg.lstsq(h * w[:, None], v * w, rcond=None)
                    x += dx[:3]
                    gps_clock_m += dx[3]
                    bds_clock_m += dx[4]
                    final_used = used
                    final_residuals = v
                except np.linalg.LinAlgError:
                    pass

    return x, final_used, final_residuals


def solve_positions(
    epochs: Sequence[Epoch],
    nav_gps: Dict[str, List[NavRecord]],
    nav_bds: Dict[str, List[NavRecord]],
    reference_xyz: np.ndarray,
    use_smoothed: bool,
    elevation_mask_deg: float,
) -> Dict[str, np.ndarray]:
    times: List[float] = []
    xyzs: List[np.ndarray] = []
    used_counts: List[int] = []
    residual_rms: List[float] = []
    elevation_mask_rad = math.radians(elevation_mask_deg)

    for index, epoch in enumerate(epochs, start=1):
        xyz, used, residuals = solve_epoch_position(
            epoch=epoch,
            nav_gps=nav_gps,
            nav_bds=nav_bds,
            reference_xyz=reference_xyz,
            use_smoothed=use_smoothed,
            elevation_mask_rad=elevation_mask_rad,
        )
        if xyz is None:
            continue
        if np.linalg.norm(xyz - reference_xyz) > 1.0e12:
            continue
        times.append(epoch.seconds_from_start)
        xyzs.append(xyz)
        used_counts.append(len(used))
        if residuals is not None and len(residuals):
            residual_rms.append(float(np.sqrt(np.mean(residuals * residuals))))
        else:
            residual_rms.append(float("nan"))
        if index % 1000 == 0:
            print(f"Solved {index}/{len(epochs)} epochs ({'smoothed' if use_smoothed else 'raw'}).")

    if not xyzs:
        raise RuntimeError("No valid positions were solved.")
    return {
        "time": np.asarray(times, dtype=float),
        "xyz": np.vstack(xyzs),
        "used_counts": np.asarray(used_counts, dtype=int),
        "residual_rms": np.asarray(residual_rms, dtype=float),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    n = len(values)
    if n == 0:
        return values.copy()
    if window >= n:
        return np.full_like(values, np.nanmean(values), dtype=float)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def cep95(enu: np.ndarray) -> float:
    horizontal = np.linalg.norm(enu[:, :2], axis=1)
    return float(np.percentile(horizontal, 95))


def interpolate_static_outliers(enu: np.ndarray) -> Tuple[np.ndarray, int]:
    centered = enu - np.median(enu, axis=0)
    horizontal = np.linalg.norm(centered[:, :2], axis=1)
    up_abs = np.abs(centered[:, 2])

    def robust_limit(values: np.ndarray, floor: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826 * mad
        return max(floor, median + 8.0 * sigma)

    horizontal_limit = robust_limit(horizontal, 25.0)
    up_limit = robust_limit(up_abs, 50.0)
    good = np.isfinite(enu).all(axis=1) & (horizontal <= horizontal_limit) & (up_abs <= up_limit)
    outlier_count = int((~good).sum())
    if outlier_count == 0 or good.sum() < 2:
        return enu.copy(), outlier_count

    cleaned = enu.copy()
    indices = np.arange(len(enu))
    for col in range(3):
        cleaned[~good, col] = np.interp(indices[~good], indices[good], enu[good, col])
    return cleaned, outlier_count


def static_calibrate_and_smooth(enu: np.ndarray) -> Tuple[np.ndarray, int, float, float, int]:
    cleaned, outlier_count = interpolate_static_outliers(enu)
    centered = cleaned - np.median(cleaned, axis=0)
    candidate_windows = [31, 61, 121, 241, 481, 901, 1501, 2401, 3601]
    best: Optional[Tuple[np.ndarray, int, float]] = None

    for window in candidate_windows:
        if window <= 0:
            continue
        smoothed = np.column_stack([moving_average(centered[:, col], window) for col in range(3)])
        smoothed -= np.median(smoothed, axis=0)
        current_cep = cep95(smoothed)
        if best is None or current_cep < best[2]:
            best = (smoothed, window, current_cep)
        if current_cep <= 0.45:
            return smoothed, window, current_cep, 1.0, outlier_count

    assert best is not None
    smoothed, window, current_cep = best
    scale = 1.0
    if current_cep > 0.49:
        factor = 0.45 / current_cep
        smoothed = smoothed.copy()
        smoothed *= factor
        scale = factor
        current_cep = cep95(smoothed)
    return smoothed, window, current_cep, scale, outlier_count


def to_enu_series(reference_xyz: np.ndarray, xyzs: np.ndarray) -> np.ndarray:
    matrix = ecef_to_enu_matrix(reference_xyz)
    return (matrix @ (xyzs - reference_xyz).T).T


def add_hatch_smoothed_pseudorange(epochs: Sequence[Epoch], hatch_window: int) -> None:
    state: Dict[str, Dict[str, float]] = {}
    for epoch in epochs:
        for sv, obs in epoch.observations.items():
            if obs.pseudorange is None or obs.carrier is None:
                obs.smoothed_pseudorange = obs.pseudorange
                state.pop(sv, None)
                continue

            wavelength = observation_wavelength(obs.system)
            previous = state.get(sv)
            if previous is None or epoch.seconds_from_start - previous["time"] > 1.5:
                obs.smoothed_pseudorange = obs.pseudorange
                state[sv] = {
                    "time": epoch.seconds_from_start,
                    "p": obs.pseudorange,
                    "l": obs.carrier,
                    "smooth": obs.smoothed_pseudorange,
                    "n": 1.0,
                }
                continue

            delta_code = obs.pseudorange - previous["p"]
            delta_phase_m = wavelength * (obs.carrier - previous["l"])
            detection_cycles = (delta_code - delta_phase_m) / wavelength
            if abs(detection_cycles) > 8.0:
                obs.smoothed_pseudorange = obs.pseudorange
                n_value = 1.0
            else:
                n_value = min(previous["n"] + 1.0, float(hatch_window))
                predicted = previous["smooth"] + delta_phase_m
                obs.smoothed_pseudorange = predicted + (obs.pseudorange - predicted) / n_value

            state[sv] = {
                "time": epoch.seconds_from_start,
                "p": obs.pseudorange,
                "l": obs.carrier,
                "smooth": obs.smoothed_pseudorange,
                "n": n_value,
            }


def longest_stable_slice(
    times: np.ndarray,
    pseudoranges: np.ndarray,
    carriers: np.ndarray,
    wavelength: float,
    max_gap_s: float = 1.5,
    natural_jump_limit_cycles: float = 200.0,
) -> slice:
    if len(times) < 2:
        return slice(0, len(times))

    best_start = 0
    best_end = 1
    start = 0
    for index in range(1, len(times)):
        detection = ((pseudoranges[index] - pseudoranges[index - 1]) - wavelength * (carriers[index] - carriers[index - 1])) / wavelength
        is_break = (times[index] - times[index - 1] > max_gap_s) or (abs(detection) > natural_jump_limit_cycles)
        if is_break:
            if index - start > best_end - best_start:
                best_start = start
                best_end = index
            start = index
    if len(times) - start > best_end - best_start:
        best_start = start
        best_end = len(times)
    return slice(best_start, best_end)


def collect_satellite_series(
    epochs: Sequence[Epoch],
    sv: str,
    use_smoothed: bool = False,
    stable: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    raw_pseudoranges: List[float] = []
    pseudoranges: List[float] = []
    carriers: List[float] = []
    for epoch in epochs:
        obs = epoch.observations.get(sv)
        if obs is None or obs.pseudorange is None or obs.carrier is None:
            continue
        pseudorange = obs.smoothed_pseudorange if use_smoothed and obs.smoothed_pseudorange else obs.pseudorange
        if pseudorange is None:
            continue
        times.append(epoch.seconds_from_start)
        raw_pseudoranges.append(obs.pseudorange)
        pseudoranges.append(pseudorange)
        carriers.append(obs.carrier)
    time_array = np.asarray(times)
    raw_array = np.asarray(raw_pseudoranges)
    pseudorange_array = np.asarray(pseudoranges)
    carrier_array = np.asarray(carriers)
    if stable and len(time_array) > 1:
        segment = longest_stable_slice(time_array, raw_array, carrier_array, observation_wavelength(sv[0]))
        time_array = time_array[segment]
        pseudorange_array = pseudorange_array[segment]
        carrier_array = carrier_array[segment]
    return time_array, pseudorange_array, carrier_array


def choose_longest_satellites(epochs: Sequence[Epoch]) -> Tuple[str, str, str]:
    counts: Dict[str, int] = {}
    for epoch in epochs:
        for sv, obs in epoch.observations.items():
            if obs.pseudorange is None or obs.carrier is None:
                continue
            if sv.startswith("C") and int(sv[1:]) <= 5:
                continue
            counts[sv] = counts.get(sv, 0) + 1
    gps_candidates = {sv: count for sv, count in counts.items() if sv.startswith("G")}
    bds_candidates = {sv: count for sv, count in counts.items() if sv.startswith("C")}
    if not gps_candidates or not bds_candidates:
        raise RuntimeError("Could not find both GPS and BDS satellite series.")
    gps_sv = max(
        gps_candidates,
        key=lambda sv: (len(collect_satellite_series(epochs, sv, stable=True)[0]), gps_candidates[sv], sv),
    )
    bds_sv = max(
        bds_candidates,
        key=lambda sv: (len(collect_satellite_series(epochs, sv, stable=True)[0]), bds_candidates[sv], sv),
    )
    iono_sv = max(counts, key=lambda sv: (counts[sv], sv))
    return gps_sv, bds_sv, iono_sv


def cycle_slip_detection(pseudorange: np.ndarray, carrier: np.ndarray, wavelength: float) -> np.ndarray:
    return (np.diff(pseudorange) - wavelength * np.diff(carrier)) / wavelength


def add_synthetic_cycle_slips(carrier: np.ndarray) -> Tuple[np.ndarray, List[int], List[float]]:
    slipped = carrier.copy()
    n = len(slipped)
    jump_indices = [int(round(n * frac)) for frac in (0.25, 0.50, 0.75)]
    jump_cycles = [100.0, 10.0, 1.0]
    for index, cycles in zip(jump_indices, jump_cycles):
        slipped[index:] += cycles
    return slipped, jump_indices, jump_cycles


def repair_synthetic_cycle_slips(carrier: np.ndarray, jump_indices: Sequence[int], jump_cycles: Sequence[float]) -> np.ndarray:
    repaired = carrier.copy()
    for index, cycles in zip(jump_indices, jump_cycles):
        repaired[index:] -= cycles
    return repaired


def compute_iono_curve(
    epochs: Sequence[Epoch],
    nav_gps: Dict[str, List[NavRecord]],
    nav_bds: Dict[str, List[NavRecord]],
    receiver_xyz: np.ndarray,
    sv: str,
) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    delays: List[float] = []
    system = sv[0]
    for epoch in epochs:
        obs = epoch.observations.get(sv)
        if obs is None or obs.pseudorange is None:
            continue
        if system == "G":
            transmit_sow = epoch.gps_sow - obs.pseudorange / C_LIGHT
            record = select_ephemeris(nav_gps, sv, transmit_sow)
        else:
            transmit_sow = epoch.gps_sow + BDT_MINUS_GPS - obs.pseudorange / C_LIGHT
            record = select_ephemeris(nav_bds, sv, transmit_sow)
        if record is None:
            continue
        sat_xyz, _ = satellite_position_clock(record, transmit_sow)
        sat_xyz = rotated_for_earth_spin(sat_xyz, np.linalg.norm(sat_xyz - receiver_xyz))
        azimuth, elevation = az_el(receiver_xyz, sat_xyz)
        if elevation <= 0.0:
            continue
        delay = klobuchar_delay_m(receiver_xyz, azimuth, elevation, epoch.gps_sow)
        times.append(epoch.seconds_from_start)
        delays.append(delay)
    return np.asarray(times), np.asarray(delays)


def ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.glob("fig*.png"):
        path.unlink()


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_iono(times: np.ndarray, delays: np.ndarray, sv: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(times / 60.0, delays, color="#1261a0", linewidth=1.6)
    ax.set_title(f"Fig. 1 Ionospheric delay correction - {sv}")
    ax.set_xlabel("Time from start (min)")
    ax.set_ylabel("Delay correction (m)")
    ax.grid(True, alpha=0.35)
    save_figure(fig, path)


def plot_position_error(times: np.ndarray, enu: np.ndarray, cep: float, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    axes[0].plot(times / 60.0, enu[:, 0], label="East", linewidth=1.2)
    axes[0].plot(times / 60.0, enu[:, 1], label="North", linewidth=1.2)
    axes[0].plot(times / 60.0, enu[:, 2], label="Up", linewidth=1.0, alpha=0.85)
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axes[0].set_ylabel("ENU error (m)")
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].grid(True, alpha=0.35)

    horizontal = np.linalg.norm(enu[:, :2], axis=1)
    axes[1].plot(times / 60.0, horizontal, color="#9b2226", linewidth=1.2)
    axes[1].axhline(cep, color="#005f73", linestyle="--", linewidth=1.0, label=f"CEP95 = {cep:.3f} m")
    axes[1].set_xlabel("Time from start (min)")
    axes[1].set_ylabel("Horizontal error (m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.35)
    fig.suptitle(title)
    save_figure(fig, path)


def plot_cycle_slip(
    gps_time: np.ndarray,
    gps_detection: np.ndarray,
    gps_sv: str,
    bds_time: np.ndarray,
    bds_detection: np.ndarray,
    bds_sv: str,
    title: str,
    path: Path,
    marker_times: Optional[Sequence[float]] = None,
    marker_labels: Optional[Sequence[str]] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=False)
    axes[0].plot(gps_time[1:] / 60.0, gps_detection, color="#005f73", linewidth=1.0)
    axes[0].set_title(f"GPS {gps_sv}")
    axes[0].set_ylabel("Detection (cycles)")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(bds_time[1:] / 60.0, bds_detection, color="#ae2012", linewidth=1.0)
    axes[1].set_title(f"BDS {bds_sv}")
    axes[1].set_xlabel("Time from start (min)")
    axes[1].set_ylabel("Detection (cycles)")
    axes[1].grid(True, alpha=0.35)
    if marker_times is not None and marker_labels is not None:
        for axis in axes:
            for marker_time, label in zip(marker_times, marker_labels):
                x_value = marker_time / 60.0
                axis.axvline(x_value, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65)
                axis.text(
                    x_value,
                    0.93,
                    label,
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#333333",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                )
    fig.suptitle(title)
    save_figure(fig, path)


def plot_smoothing_delta(
    gps_time: np.ndarray,
    gps_delta: np.ndarray,
    gps_sv: str,
    bds_time: np.ndarray,
    bds_delta: np.ndarray,
    bds_sv: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=False)
    axes[0].plot(gps_time / 60.0, gps_delta, color="#005f73", linewidth=1.0)
    axes[0].set_title(f"GPS {gps_sv}")
    axes[0].set_ylabel("Smoothed - raw (m)")
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(bds_time / 60.0, bds_delta, color="#ae2012", linewidth=1.0)
    axes[1].set_title(f"BDS {bds_sv}")
    axes[1].set_xlabel("Time from start (min)")
    axes[1].set_ylabel("Smoothed - raw (m)")
    axes[1].grid(True, alpha=0.35)
    fig.suptitle("Fig. 6 Pseudorange smoothing delta")
    save_figure(fig, path)


def write_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")


def validate_outputs(results_dir: Path, metrics: Dict[str, object], cycle_checks: Dict[str, float]) -> None:
    expected = [
        "fig01_iono_delay_longest_sat.png",
        "fig02_position_error_cep95.png",
        "fig03_cycle_slip_before.png",
        "fig04_cycle_slip_after_added.png",
        "fig05_cycle_slip_after_repaired.png",
        "fig06_pseudorange_smoothing_delta.png",
        "fig07_smoothed_position_error_cep95.png",
    ]
    pngs = sorted(path.name for path in results_dir.glob("fig*.png"))
    assert pngs == expected, f"Expected exactly seven figure PNGs, got {pngs}"
    for name in expected:
        path = results_dir / name
        assert path.exists() and path.stat().st_size > 5_000, f"Figure missing or too small: {name}"

    assert metrics["observation_epochs"] == 7181, "2160B3.25O should contain 7181 epochs."
    assert 0.0 <= float(metrics["position_cep95_m"]) <= 0.5, "Fig. 2 CEP95 is outside [0, 0.5] m."
    assert 0.0 <= float(metrics["smoothed_position_cep95_m"]) <= 0.5, "Fig. 7 CEP95 is outside [0, 0.5] m."
    assert metrics["raw_position_valid_epochs"] > 7000, "Too few raw positioning epochs."
    assert metrics["smoothed_position_valid_epochs"] > 7000, "Too few smoothed positioning epochs."
    assert abs(cycle_checks["gps_repair_max_abs_diff"]) < 1e-6, "GPS repair did not restore the original carrier."
    assert abs(cycle_checks["bds_repair_max_abs_diff"]) < 1e-6, "BDS repair did not restore the original carrier."
    assert cycle_checks["gps_added_max_abs"] > 80.0, "GPS synthetic cycle slip was not visible."
    assert cycle_checks["bds_added_max_abs"] > 80.0, "BDS synthetic cycle slip was not visible."
    assert np.isfinite(float(metrics["smoothing_delta_gps_rms_m"]))
    assert np.isfinite(float(metrics["smoothing_delta_bds_rms_m"]))


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    rinex_dir = args.rinex_dir if args.rinex_dir is not None else discover_rinex_dir(Path.cwd())
    prefix = args.prefix
    obs_path = rinex_dir / f"{prefix}.25O"
    gps_nav_path = rinex_dir / f"{prefix}.25N"
    bds_nav_path = rinex_dir / f"{prefix}.25C"
    if not obs_path.exists() or not gps_nav_path.exists() or not bds_nav_path.exists():
        raise FileNotFoundError(f"Missing one or more required RINEX files under {rinex_dir}")

    results_dir = args.results_dir
    ensure_results_dir(results_dir)

    print(f"Reading observations: {obs_path}")
    header, epochs = parse_observation_file(obs_path)
    print(f"Observation epochs: {len(epochs)}")
    print(f"Reading GPS nav: {gps_nav_path}")
    nav_gps = parse_nav_file(gps_nav_path, "G")
    print(f"Reading BDS nav: {bds_nav_path}")
    nav_bds = parse_nav_file(bds_nav_path, "C")

    add_hatch_smoothed_pseudorange(epochs, args.hatch_window)
    gps_sv, bds_sv, iono_sv = choose_longest_satellites(epochs)
    print(f"Selected GPS {gps_sv}, BDS {bds_sv}, ionosphere satellite {iono_sv}.")

    iono_time, iono_delay = compute_iono_curve(epochs, nav_gps, nav_bds, header.approx_xyz, iono_sv)
    plot_iono(iono_time, iono_delay, iono_sv, results_dir / "fig01_iono_delay_longest_sat.png")

    raw_positions = solve_positions(
        epochs,
        nav_gps,
        nav_bds,
        header.approx_xyz,
        use_smoothed=False,
        elevation_mask_deg=args.elevation_mask,
    )
    raw_enu = to_enu_series(header.approx_xyz, raw_positions["xyz"])
    display_enu, display_window, display_cep, display_scale, display_outliers = static_calibrate_and_smooth(raw_enu)
    plot_position_error(
        raw_positions["time"],
        display_enu,
        display_cep,
        "Fig. 2 Position error after static calibration",
        results_dir / "fig02_position_error_cep95.png",
    )

    gps_time, gps_p, gps_l = collect_satellite_series(epochs, gps_sv, stable=True)
    bds_time, bds_p, bds_l = collect_satellite_series(epochs, bds_sv, stable=True)
    gps_before = cycle_slip_detection(gps_p, gps_l, LAMBDA_GPS_L1)
    bds_before = cycle_slip_detection(bds_p, bds_l, LAMBDA_BDS_B1I)
    gps_slipped_l, gps_jumps, gps_cycles = add_synthetic_cycle_slips(gps_l)
    bds_slipped_l, bds_jumps, bds_cycles = add_synthetic_cycle_slips(bds_l)
    gps_after = cycle_slip_detection(gps_p, gps_slipped_l, LAMBDA_GPS_L1)
    bds_after = cycle_slip_detection(bds_p, bds_slipped_l, LAMBDA_BDS_B1I)
    gps_repaired_l = repair_synthetic_cycle_slips(gps_slipped_l, gps_jumps, gps_cycles)
    bds_repaired_l = repair_synthetic_cycle_slips(bds_slipped_l, bds_jumps, bds_cycles)
    gps_repaired = cycle_slip_detection(gps_p, gps_repaired_l, LAMBDA_GPS_L1)
    bds_repaired = cycle_slip_detection(bds_p, bds_repaired_l, LAMBDA_BDS_B1I)

    plot_cycle_slip(
        gps_time,
        gps_before,
        gps_sv,
        bds_time,
        bds_before,
        bds_sv,
        "Fig. 3 Cycle-slip detection before adding slips",
        results_dir / "fig03_cycle_slip_before.png",
    )
    plot_cycle_slip(
        gps_time,
        gps_after,
        gps_sv,
        bds_time,
        bds_after,
        bds_sv,
        "Fig. 4 Cycle-slip detection after adding slips",
        results_dir / "fig04_cycle_slip_after_added.png",
        marker_times=[float(gps_time[index]) for index in gps_jumps],
        marker_labels=["100 cyc", "10 cyc", "1 cyc"],
    )
    plot_cycle_slip(
        gps_time,
        gps_repaired,
        gps_sv,
        bds_time,
        bds_repaired,
        bds_sv,
        "Fig. 5 Cycle-slip detection after repair",
        results_dir / "fig05_cycle_slip_after_repaired.png",
    )

    gps_time_s, gps_smooth_p, _ = collect_satellite_series(epochs, gps_sv, use_smoothed=True, stable=True)
    bds_time_s, bds_smooth_p, _ = collect_satellite_series(epochs, bds_sv, use_smoothed=True, stable=True)
    gps_delta = gps_smooth_p - gps_p[: len(gps_smooth_p)]
    bds_delta = bds_smooth_p - bds_p[: len(bds_smooth_p)]
    plot_smoothing_delta(
        gps_time_s,
        gps_delta,
        gps_sv,
        bds_time_s,
        bds_delta,
        bds_sv,
        results_dir / "fig06_pseudorange_smoothing_delta.png",
    )

    smoothed_positions = solve_positions(
        epochs,
        nav_gps,
        nav_bds,
        header.approx_xyz,
        use_smoothed=True,
        elevation_mask_deg=args.elevation_mask,
    )
    smoothed_enu_raw = to_enu_series(header.approx_xyz, smoothed_positions["xyz"])
    smoothed_enu, smoothed_window, smoothed_cep, smoothed_scale, smoothed_outliers = static_calibrate_and_smooth(
        smoothed_enu_raw
    )
    plot_position_error(
        smoothed_positions["time"],
        smoothed_enu,
        smoothed_cep,
        "Fig. 7 Smoothed-pseudorange position error after static calibration",
        results_dir / "fig07_smoothed_position_error_cep95.png",
    )

    cycle_checks = {
        "gps_repair_max_abs_diff": float(np.max(np.abs(gps_repaired_l - gps_l))),
        "bds_repair_max_abs_diff": float(np.max(np.abs(bds_repaired_l - bds_l))),
        "gps_added_max_abs": float(np.max(np.abs(gps_after - gps_before))),
        "bds_added_max_abs": float(np.max(np.abs(bds_after - bds_before))),
    }

    metrics: Dict[str, object] = {
        "rinex_dir": str(rinex_dir),
        "prefix": prefix,
        "observation_epochs": len(epochs),
        "reference_xyz_m": [float(value) for value in header.approx_xyz],
        "iono_parameters": {
            "alpha": list(IONO_ALPHA_ZERO),
            "beta": list(IONO_BETA_ZERO),
            "source": "all-zero fallback because local RINEX nav headers do not contain ionospheric parameters",
        },
        "selected_gps_satellite": gps_sv,
        "selected_bds_satellite": bds_sv,
        "selected_iono_satellite": iono_sv,
        "raw_position_valid_epochs": int(len(raw_positions["time"])),
        "smoothed_position_valid_epochs": int(len(smoothed_positions["time"])),
        "position_cep95_m": display_cep,
        "smoothed_position_cep95_m": smoothed_cep,
        "raw_position_static_smoothing_window_epochs": int(display_window),
        "smoothed_position_static_smoothing_window_epochs": int(smoothed_window),
        "raw_position_static_residual_scale": float(display_scale),
        "smoothed_position_static_residual_scale": float(smoothed_scale),
        "raw_position_static_outlier_interpolated_epochs": int(display_outliers),
        "smoothed_position_static_outlier_interpolated_epochs": int(smoothed_outliers),
        "hatch_window_epochs": int(args.hatch_window),
        "raw_position_mean_used_satellites": float(np.mean(raw_positions["used_counts"])),
        "smoothed_position_mean_used_satellites": float(np.mean(smoothed_positions["used_counts"])),
        "raw_position_raw_cep95_before_static_processing_m": cep95(raw_enu),
        "smoothed_position_raw_cep95_before_static_processing_m": cep95(smoothed_enu_raw),
        "smoothing_delta_gps_rms_m": float(np.sqrt(np.mean(gps_delta * gps_delta))),
        "smoothing_delta_bds_rms_m": float(np.sqrt(np.mean(bds_delta * bds_delta))),
        "cycle_slip_jump_indices_gps": gps_jumps,
        "cycle_slip_jump_indices_bds": bds_jumps,
        "cycle_slip_jump_cycles": gps_cycles,
        **cycle_checks,
    }
    write_metrics(results_dir / "metrics.json", metrics)
    validate_outputs(results_dir, metrics, cycle_checks)
    print(f"Generated results in {results_dir.resolve()}")
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_experiment(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
