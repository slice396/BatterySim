from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List


@dataclass
class BatterySpec:
    name: str
    usable_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    roundtrip_efficiency: float
    purchase_price_eur: float


PRESET_BATTERIES: List[BatterySpec] = [
    BatterySpec("Zendure Hyper 2000 + 7,68 kWh", 7.68, 2.0, 2.0, 0.90, 4299),
    BatterySpec("Huawei LUNA2000 10 kWh", 10.0, 5.0, 5.0, 0.93, 6999),
    BatterySpec("BYD Battery-Box Premium HVS 10.2", 10.2, 5.0, 5.0, 0.95, 7499),
    BatterySpec("Sessy thuisbatterij 5 kWh", 5.0, 1.7, 1.7, 0.90, 3650),
    BatterySpec("HomeWizard PIB 1 st", 2.7, 0.8, 0.8, 0.90, 1150),
    BatterySpec("HomeWizard PIB 2 st", 5.4, 1.6, 1.6, 0.90, 2300),
    BatterySpec("HomeWizard PIB 3 st", 8.1, 2.4, 2.4, 0.90, 3450),
    BatterySpec("Marstek Venus 3.0", 5.12, 2.5, 2.5, 0.90, 1300),
    BatterySpec("Tesla Powerwall 3 (~13,5 kWh)", 13.5, 5.0, 5.0, 0.90, 9499),
]

CUSTOM_BATTERY_FILE = Path(__file__).resolve().with_name("battery_models.custom.json")


def load_all_batteries() -> List[BatterySpec]:
    batteries = [BatterySpec(**vars(b)) for b in PRESET_BATTERIES]
    if not CUSTOM_BATTERY_FILE.exists():
        return batteries
    try:
        raw = json.loads(CUSTOM_BATTERY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return batteries
    if not isinstance(raw, list):
        return batteries
    for item in raw:
        try:
            batteries.append(BatterySpec(**item))
        except Exception:
            continue
    return batteries


def save_custom_batteries(all_batteries: List[BatterySpec]) -> None:
    preset_names = {battery.name for battery in PRESET_BATTERIES}
    custom = [asdict(battery) for battery in all_batteries if battery.name not in preset_names]
    CUSTOM_BATTERY_FILE.write_text(json.dumps(custom, indent=2, ensure_ascii=False), encoding="utf-8")
