def test_dataset_config_option_and_fixtures_are_registered():
    import parity.conftest as conftest_module

    assert hasattr(conftest_module, "pytest_addoption")
    assert hasattr(conftest_module, "active_profile")
    assert hasattr(conftest_module, "indexed_pyqmd_store")
