"""Smoke tests for the harness scaffold and its configuration invariants."""

from __future__ import annotations

import airm
from airm import config


def test_package_imports():
    assert airm.__version__


def test_repo_root_points_at_the_checkout():
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    assert (config.REPO_ROOT / "EXPERIMENTS.md").exists()


def test_six_formats_with_json_as_baseline():
    assert len(config.FORMATS) == 6
    assert len(set(config.FORMATS)) == 6
    assert config.BASELINE_FORMAT in config.FORMATS
    assert set(config.FORMAT_LABELS) == set(config.FORMATS)


def test_topics_partition_into_primary_and_secondary():
    assert set(config.PRIMARY_TOPICS) | set(config.SECONDARY_TOPICS) == set(config.CMR_TOPICS)
    assert not set(config.PRIMARY_TOPICS) & set(config.SECONDARY_TOPICS)
    assert len(config.PRIMARY_TOPICS) == 3


def test_synthetic_query_quota_sums_to_the_declared_total():
    quota = config.synthetic_quota()
    assert sum(quota.values()) == config.SYNTHETIC_QUERY_TOTAL
    assert set(quota) == set(config.CMR_TOPICS)


def test_synthetic_quota_gives_the_three_required_domains_equal_shares():
    quota = config.synthetic_quota()
    assert len({quota[t] for t in config.PRIMARY_TOPICS}) == 1
    assert quota[config.PRIMARY_TOPICS[0]] == config.SYNTHETIC_PER_PRIMARY_TOPIC


def test_every_other_domain_still_gets_representation():
    quota = config.synthetic_quota()
    assert all(quota[t] >= 2 for t in config.SECONDARY_TOPICS)


def test_judge_is_not_drawn_from_the_system_under_test_matrix():
    """A judge that varies with the tested model confounds the model axis."""
    assert config.JUDGE_MODEL
    assert config.JUDGE_PROVIDER == "openai"


def test_collection_names_are_distinct_per_format():
    names = {config.collection_name(f) for f in config.FORMATS}
    assert len(names) == len(config.FORMATS)
