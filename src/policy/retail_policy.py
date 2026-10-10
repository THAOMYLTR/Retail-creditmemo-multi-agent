"""Loader for the retail product policy (src/policy/retail_products.yaml).

The calculators never hard-code a DSR cap or an LTV: they ask this module for
the limits of the product the request named. When no product was named, the
*strictest* value of each limit across all products is used, mirroring how the
document matrix falls back to the strongest R/O level — an unnamed product can
only make the result more conservative, never quietly looser.

Validated at load time and raises PolicyError on the first problem, because a
silently mis-parsed cap would turn into a wrong loan amount with nothing in the
output pointing at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

POLICY_PATH = Path(__file__).resolve().parent / "retail_products.yaml"

# Limits where a LOWER value is stricter (a cap) versus HIGHER is stricter
# (a floor). Used to pick the conservative value when no product is named.
_CAP_FIELDS = (
    "max_dsr",
    "max_ltv",
    "max_tenor_months",
    "max_age_at_maturity",
    "max_amount",
    "max_income_multiple",
)
_FLOOR_FIELDS = ("min_own_contribution", "min_monthly_income")
# Limits that only make sense for the product that defines them.
_PRODUCT_SPECIFIC_FIELDS = ("max_amount", "max_income_multiple")
PRODUCT_FIELDS = _CAP_FIELDS + _FLOOR_FIELDS

INCOME_SOURCE_TYPES = ("luong", "huu_tri", "cho_thue", "kinh_doanh", "khac")


class PolicyError(RuntimeError):
    """Raised when the policy file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ProductLimits:
    """The limits applied to one loan, and which product they came from."""

    product_id: str
    label: str
    max_dsr: float | None = None
    max_ltv: float | None = None
    min_own_contribution: float | None = None
    max_tenor_months: float | None = None
    max_age_at_maturity: float | None = None
    max_amount: float | None = None
    max_income_multiple: float | None = None
    min_monthly_income: float | None = None


@dataclass(frozen=True)
class RetailPolicy:
    version: int
    defaults: dict[str, Any]
    products: dict[str, ProductLimits] = field(default_factory=dict)

    def income_recognition(self, source_type: str) -> float:
        table = self.defaults.get("income_recognition") or {}
        return float(table.get(source_type, table.get("khac", 0.5)))

    def limits_for(self, product_id: str | None) -> ProductLimits:
        """Limits for one product, or the strictest combination of all.

        The strictest combination leaves out the product-specific caps (a flat
        maximum amount, an income multiple): those belong to one product and are
        not a stricter version of anything the others have.
        """

        if product_id and product_id in self.products:
            return self.products[product_id]
        if not self.products:
            return ProductLimits(product_id="", label="(không có chính sách)")
        values: dict[str, float | None] = {}
        for name in PRODUCT_FIELDS:
            if name in _PRODUCT_SPECIFIC_FIELDS:
                # A flat amount cap or an income multiple describes one product
                # (the unsecured loan), not a general strictness level. Applied
                # to an unknown product it would cap a mortgage at the consumer
                # loan's ceiling and report that as the binding limit.
                values[name] = None
                continue
            present = [
                getattr(limits, name)
                for limits in self.products.values()
                if getattr(limits, name) is not None
            ]
            if not present:
                values[name] = None
            elif name in _CAP_FIELDS:
                values[name] = min(present)
            else:
                values[name] = max(present)
        return ProductLimits(
            product_id="",
            label="Chưa xác định sản phẩm — áp dụng mức chặt nhất",
            **values,
        )


def _fail(message: str) -> None:
    raise PolicyError(f"{POLICY_PATH.name}: {message}")


def _ratio(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected a number or null, got {value!r}")
    if not 0 <= float(value) <= 1:
        _fail(f"{where}: ratio {value!r} must be between 0 and 1")
    return float(value)


def _positive(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected a number or null, got {value!r}")
    if float(value) <= 0:
        _fail(f"{where}: {value!r} must be positive")
    return float(value)


def _load(path: Path) -> RetailPolicy:
    if not path.exists():
        _fail(f"policy file not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        _fail("top level must be a mapping")
    version = raw.get("version")
    if not isinstance(version, int):
        _fail("'version' must be an integer")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        _fail("'defaults' must be a mapping")
    recognition = defaults.get("income_recognition") or {}
    if not isinstance(recognition, dict):
        _fail("'defaults.income_recognition' must be a mapping")
    for source_type, value in recognition.items():
        if source_type not in INCOME_SOURCE_TYPES:
            _fail(
                f"income_recognition: unknown source type {source_type!r}; "
                f"expected one of {list(INCOME_SOURCE_TYPES)}"
            )
        _ratio(value, f"income_recognition.{source_type}")
    _ratio(defaults.get("card_payment_rate"), "defaults.card_payment_rate")
    _ratio(defaults.get("income_drop_alert"), "defaults.income_drop_alert")

    raw_products = raw.get("products") or {}
    if not isinstance(raw_products, dict) or not raw_products:
        _fail("'products' must be a non-empty mapping")
    products: dict[str, ProductLimits] = {}
    for product_id, entry in raw_products.items():
        where = f"product {product_id!r}"
        if not isinstance(entry, dict):
            _fail(f"{where}: must be a mapping")
        unknown = set(entry) - set(PRODUCT_FIELDS) - {"label"}
        if unknown:
            _fail(f"{where}: unknown field(s) {sorted(unknown)}")
        values: dict[str, float | None] = {}
        for name in ("max_dsr", "max_ltv", "min_own_contribution"):
            values[name] = _ratio(entry.get(name), f"{where}.{name}")
        for name in (
            "max_tenor_months",
            "max_age_at_maturity",
            "max_amount",
            "max_income_multiple",
            "min_monthly_income",
        ):
            values[name] = _positive(entry.get(name), f"{where}.{name}")
        products[str(product_id)] = ProductLimits(
            product_id=str(product_id),
            label=str(entry.get("label") or product_id),
            **values,
        )
    return RetailPolicy(version=version, defaults=defaults, products=products)


@lru_cache(maxsize=1)
def load_policy() -> RetailPolicy:
    """Load and validate the policy once per process."""

    return _load(POLICY_PATH)


def reload_policy() -> RetailPolicy:
    """Drop the cache and re-read the file (for tests / notebook iteration)."""

    load_policy.cache_clear()
    return load_policy()
