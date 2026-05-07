def test_feature_package_exports_skip_helpers_after_merge():
    import feature

    assert feature.SKIP_TOKENS == {"a", "an", "the", "of", ".", ",", ""}
    assert callable(feature.token_should_be_skipped)
