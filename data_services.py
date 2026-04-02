from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DEFAULT_PRICE_IMPORT = 0.32
DEFAULT_PRICE_EXPORT = 0.08
DEFAULT_FRANK_OPSLAG = 0.02
DEFAULT_ENTSOE_ZONE = "10YNL----------L"
INTERVAL_HOURS = 0.25
ENTSOE_API_URLS = ["https://web-api.tp.entsoe.eu/api", "https://web-api.tp.entsoe.eu/api/"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in df.columns]
    return out


def to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace("\u00a0", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def is_homewizard_p1_export(df: pd.DataFrame) -> bool:
    required = {"time", "Import T1 kWh", "Import T2 kWh", "Export T1 kWh", "Export T2 kWh"}
    return required.issubset(set(df.columns))


def prepare_homewizard_p1_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy().sort_values("time")
    for col in ["Import T1 kWh", "Import T2 kWh", "Export T1 kWh", "Export T2 kWh"]:
        df[col] = to_numeric_series(df[col])
    df["grid_import_kwh"] = df["Import T1 kWh"].diff().fillna(0).clip(lower=0) + df["Import T2 kWh"].diff().fillna(0).clip(lower=0)
    df["grid_export_kwh"] = df["Export T1 kWh"].diff().fillna(0).clip(lower=0) + df["Export T2 kWh"].diff().fillna(0).clip(lower=0)
    prepared = pd.DataFrame({
        "timestamp": df["time"],
        "consumption_kwh": df["grid_import_kwh"],
        "production_kwh": df["grid_export_kwh"],
        "direct_self_consumption_kwh": 0.0,
        "surplus_kwh": df["grid_export_kwh"],
        "deficit_kwh": df["grid_import_kwh"],
    }).reset_index(drop=True)
    prepared.attrs["data_mode"] = "homewizard_p1"
    return prepared


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["timestamp", "datetime", "date", "tijdstip", "tijd", "time", "local_datetime", "starts_at", "from"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.8:
                return col
        except Exception:
            pass
    return None


def find_numeric_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for col_lower, original in lower_cols.items():
        if any(k in col_lower for k in keywords):
            coerced = to_numeric_series(df[original])
            if coerced.notna().mean() > 0.5:
                return original
    return None


def infer_energy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    consumption_col = find_numeric_column(df, ["consumption", "verbruik", "usage", "afname", "load", "import"])
    production_col = find_numeric_column(df, ["production", "opwek", "generation", "pv", "solar", "yield", "export"])
    net_col = find_numeric_column(df, ["net", "balance", "saldo", "grid", "vermogen", "power"])
    return consumption_col, production_col, net_col


def prepare_energy_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)
    if is_homewizard_p1_export(df):
        return prepare_homewizard_p1_dataframe(df)
    ts_col = find_timestamp_column(df)
    if not ts_col:
        raise ValueError("Geen tijdkolom gevonden.")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[ts_col]).copy().sort_values(ts_col).rename(columns={ts_col: "timestamp"})
    consumption_col, production_col, net_col = infer_energy_columns(df)
    for col in [consumption_col, production_col, net_col]:
        if col:
            df[col] = to_numeric_series(df[col])
    if consumption_col and production_col:
        df["consumption_kwh"] = df[consumption_col].fillna(0).clip(lower=0)
        df["production_kwh"] = df[production_col].fillna(0).clip(lower=0)
    elif net_col:
        net = df[net_col].fillna(0)
        df["consumption_kwh"] = net.clip(lower=0)
        df["production_kwh"] = (-net.clip(upper=0))
    else:
        raise ValueError("Geen bruikbare verbruiks-, opwek- of netto-kolom gevonden.")
    max_val = max(df["consumption_kwh"].max(), df["production_kwh"].max())
    if max_val > 50:
        df["consumption_kwh"] = df["consumption_kwh"] / 1000 * INTERVAL_HOURS
        df["production_kwh"] = df["production_kwh"] / 1000 * INTERVAL_HOURS
    df["direct_self_consumption_kwh"] = np.minimum(df["consumption_kwh"], df["production_kwh"])
    df["surplus_kwh"] = (df["production_kwh"] - df["consumption_kwh"]).clip(lower=0)
    df["deficit_kwh"] = (df["consumption_kwh"] - df["production_kwh"]).clip(lower=0)
    prepared = df[["timestamp", "consumption_kwh", "production_kwh", "direct_self_consumption_kwh", "surplus_kwh", "deficit_kwh"]].reset_index(drop=True)
    prepared.attrs["data_mode"] = "generic"
    return prepared


def load_energy_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")
    return prepare_energy_dataframe(raw)


def prepare_price_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)
    ts_col = find_timestamp_column(df)
    if not ts_col:
        raise ValueError("Geen tijdkolom gevonden in prijs-CSV.")
    import_col = find_numeric_column(df, ["import", "afname", "verbruik", "buy", "purchase", "price", "prijs"])
    export_col = find_numeric_column(df, ["export", "teruglever", "sell", "feed", "inject"])
    if not import_col:
        raise ValueError("Geen prijs-kolom voor afname gevonden in prijs-CSV.")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[ts_col]).copy().sort_values(ts_col)
    df[import_col] = to_numeric_series(df[import_col])
    if export_col:
        df[export_col] = to_numeric_series(df[export_col])
    else:
        df["__export_price__"] = np.nan
        export_col = "__export_price__"
    price_df = pd.DataFrame({
        "timestamp": df[ts_col],
        "import_price_eur_per_kwh": df[import_col],
        "export_price_eur_per_kwh": df[export_col],
    }).dropna(subset=["import_price_eur_per_kwh"])
    if price_df["export_price_eur_per_kwh"].isna().all():
        price_df["export_price_eur_per_kwh"] = np.nan
    return price_df.reset_index(drop=True)


def load_price_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")
    return prepare_price_dataframe(raw)


def floor_to_day_utc(timestamp: pd.Timestamp) -> datetime:
    dt = timestamp.tz_localize(None).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(tzinfo=timezone.utc)


def ceil_to_next_day_utc(timestamp: pd.Timestamp) -> datetime:
    dt = timestamp.tz_localize(None).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return dt.replace(tzinfo=timezone.utc)


def fetch_entsoe_day_ahead_prices(zone_eic: str, start_dt: datetime, end_dt: datetime, token: str) -> pd.DataFrame:
    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": zone_eic,
        "out_Domain": zone_eic,
        "periodStart": start_dt.strftime("%Y%m%d%H%M"),
        "periodEnd": end_dt.strftime("%Y%m%d%H%M"),
    }
    xml_data = b""
    last_error = "Onbekende fout"

    for base_url in ENTSOE_API_URLS:
        url = f"{base_url}?{urlencode(params)}"
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "BatterySim/1.0", "Accept": "application/xml,text/xml,*/*"})
                with urlopen(request, timeout=60) as response:
                    xml_data = response.read()
                break
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if exc.code in {503, 504} and attempt < 2:
                    continue
                if exc.code == 404 and "<html" in body.lower():
                    last_error = (
                        "HTTP 404 op ENTSO-E endpoint. Probeer zowel /api als /api/; "
                        "de server geeft nu geen geldige API-response terug."
                    )
                    # probeer volgende endpointvariant (met/zonder trailing slash)
                    break
                if exc.code == 401:
                    last_error = "HTTP 401: Ongeldige of ontbrekende ENTSO-E securityToken."
                else:
                    last_error = f"HTTP {exc.code}: {body[:500]}"
                break
            except URLError as exc:
                last_error = f"Niet bereikbaar: {exc}"
                break
        if xml_data:
            break

    if not xml_data:
        raise ValueError(
            "ENTSO-E API fout. De server was niet beschikbaar of gaf een ongeldige reactie. "
            f"Laatste melding: {last_error}"
        )

    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    root = ET.fromstring(xml_data)
    rows = []
    for ts in root.findall(".//ns:TimeSeries", ns):
        period = ts.find("ns:Period", ns)
        if period is None:
            continue
        start_text = period.findtext("ns:timeInterval/ns:start", default="", namespaces=ns)
        resolution_text = period.findtext("ns:resolution", default="PT60M", namespaces=ns)
        if not start_text:
            continue
        period_start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        step = timedelta(minutes=15) if resolution_text == "PT15M" else timedelta(minutes=30) if resolution_text == "PT30M" else timedelta(hours=1)
        for point in period.findall("ns:Point", ns):
            position = point.findtext("ns:position", default="0", namespaces=ns)
            price = point.findtext("ns:price.amount", default="", namespaces=ns)
            if not price:
                continue
            ts_dt = period_start + (int(position) - 1) * step
            rows.append({"timestamp": pd.Timestamp(ts_dt).tz_convert(None), "import_price_eur_per_mwh": float(price)})
    if not rows:
        raise ValueError("Geen ENTSO-E prijsdata ontvangen voor de gekozen periode.")
    out = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out["import_price_eur_per_kwh"] = out["import_price_eur_per_mwh"] / 1000.0
    out["export_price_eur_per_kwh"] = np.nan
    return out[["timestamp", "import_price_eur_per_kwh", "export_price_eur_per_kwh"]].reset_index(drop=True)


def fetch_entsoe_day_ahead_prices_chunked(zone_eic: str, start_dt: datetime, end_dt: datetime, token: str) -> pd.DataFrame:
    max_span = timedelta(days=365)
    chunk_start = start_dt
    frames = []
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + max_span, end_dt)
        frames.append(fetch_entsoe_day_ahead_prices(zone_eic, chunk_start, chunk_end, token))
        chunk_start = chunk_end
    if not frames:
        raise ValueError("Geen ENTSO-E prijsdata opgehaald.")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def download_entsoe_prices_for_period(energy_df: pd.DataFrame, zone_eic: str, token: str) -> pd.DataFrame:
    start_dt = floor_to_day_utc(energy_df["timestamp"].min())
    end_dt = ceil_to_next_day_utc(energy_df["timestamp"].max())
    return fetch_entsoe_day_ahead_prices_chunked(zone_eic, start_dt, end_dt, token)


def align_prices_to_energy(energy_df: pd.DataFrame, price_df: Optional[pd.DataFrame], mode: str, fixed_import: float, fixed_export: float, opslag: float) -> pd.DataFrame:
    if mode == "fixed":
        out = energy_df.copy()
        out["import_price_eur_per_kwh"] = fixed_import + opslag
        out["export_price_eur_per_kwh"] = fixed_export
        out.attrs["price_mode"] = "fixed"
        out.attrs["frank_opslag"] = opslag
        out.attrs["price_overlap_ratio"] = 1.0
        return out
    if price_df is None or price_df.empty:
        raise ValueError("Voor deze prijsmode is prijsdata nodig, maar die is nog niet geladen.")
    energy = energy_df.sort_values("timestamp").copy()
    prices = price_df.sort_values("timestamp").copy()
    if mode == "entsoe_api":
        seconds = prices["timestamp"].diff().dropna().dt.total_seconds()
        if not seconds.empty and seconds.median() >= 3599:
            expanded_rows = []
            for _, row in prices.iterrows():
                for minutes in (0, 15, 30, 45):
                    expanded_rows.append({
                        "timestamp": row["timestamp"] + pd.Timedelta(minutes=minutes),
                        "import_price_eur_per_kwh": row["import_price_eur_per_kwh"],
                        "export_price_eur_per_kwh": row["export_price_eur_per_kwh"],
                    })
            prices = pd.DataFrame(expanded_rows).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    tolerance = pd.Timedelta("1h") if mode == "dynamic_csv" else pd.Timedelta("20m")
    aligned = pd.merge_asof(energy, prices, on="timestamp", direction="backward", tolerance=tolerance)
    overlap_mask = aligned["import_price_eur_per_kwh"].notna()
    overlap = overlap_mask.mean() if len(aligned) else 0.0
    aligned["import_price_eur_per_kwh"] = aligned["import_price_eur_per_kwh"].fillna(fixed_import) + opslag
    aligned["export_price_eur_per_kwh"] = aligned["export_price_eur_per_kwh"].fillna(fixed_export)
    aligned.attrs["price_mode"] = mode
    aligned.attrs["price_overlap_ratio"] = overlap
    aligned.attrs["frank_opslag"] = opslag
    return aligned
