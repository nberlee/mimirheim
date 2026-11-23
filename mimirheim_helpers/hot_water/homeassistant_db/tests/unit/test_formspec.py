"""Tests for the FormSpec authored for HotWaterDrawConfig.

Verifies the FormSpec stays aligned with the config model at every depth
(ADR-0006), builds into a Config Service Descriptor, represents both tiers,
and that its Conditional Visibility rules evaluate as expected.
"""

from mimirheim_shared.alignment import assert_form_spec_complete
from mimirheim_shared.config_service import build_descriptor
from mimirheim_shared.formspec import resolve_field_shapes
from mimirheim_shared.visibility import evaluate_condition

from hot_water_draw.config import HotWaterDrawConfig
from hot_water_draw.formspec import HOT_WATER_DRAW_CONFIG_FORM_SPEC


def test_form_spec_is_aligned_with_config() -> None:
    assert_form_spec_complete(HotWaterDrawConfig, HOT_WATER_DRAW_CONFIG_FORM_SPEC)


def test_form_spec_builds_into_a_descriptor() -> None:
    descriptor = build_descriptor(
        "hot_water_draw", "Hot water draw", HotWaterDrawConfig, HOT_WATER_DRAW_CONFIG_FORM_SPEC
    )
    assert descriptor.owner_id == "hot_water_draw"
    assert descriptor.form_spec == resolve_field_shapes(HotWaterDrawConfig, HOT_WATER_DRAW_CONFIG_FORM_SPEC)
    assert descriptor.json_schema == HotWaterDrawConfig.model_json_schema()


def test_form_spec_has_both_tiers_represented() -> None:
    tiers = {field.tier for field in HOT_WATER_DRAW_CONFIG_FORM_SPEC.fields.values()}
    assert len(tiers) == 2


def test_mimir_trigger_topic_is_only_visible_when_signal_mimir_is_enabled() -> None:
    condition = HOT_WATER_DRAW_CONFIG_FORM_SPEC.fields["mimir_trigger_topic"].visible_if
    assert condition is not None
    assert evaluate_condition(condition, {"signal_mimir": True}) is True
    assert evaluate_condition(condition, {"signal_mimir": False}) is False


def test_both_tank_loss_fields_stay_visible() -> None:
    """Hiding one would strand a stored value: the editor cannot clear a hidden field."""
    tank = HOT_WATER_DRAW_CONFIG_FORM_SPEC.fields["tank"].nested_form_spec
    assert tank is not None
    for name in ("standby_loss_w", "standby_loss_kw_per_k"):
        assert tank.fields[name].visible_if is None
        assert "exactly one" in tank.fields[name].description
