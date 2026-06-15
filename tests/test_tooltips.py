import tooltips

REQUIRED_KEYS = {
    "prompt",
    "image",
    "target_image",
    "ref_image",
    "speed",
    "steps",
    "true_cfg",
    "negative_prompt",
    "seed",
}


def test_tooltips_has_all_required_keys():
    assert REQUIRED_KEYS <= set(tooltips.TOOLTIPS)


def test_tooltips_values_are_non_empty_strings():
    for key, val in tooltips.TOOLTIPS.items():
        assert isinstance(val, str) and val.strip(), f"{key} is empty or non-string"


def test_tooltips_values_are_short_enough_for_a_tooltip():
    for key, val in tooltips.TOOLTIPS.items():
        assert len(val) <= 200, f"{key} is too long for a tooltip ({len(val)} chars)"
