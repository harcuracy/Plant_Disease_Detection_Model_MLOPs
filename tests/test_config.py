from cnnClassifier.config.configuration import ConfigurationManager


def test_configuration_manager_loads_configs():
    config = ConfigurationManager()

    data_config = config.get_data_ingestion_config()
    training_config = config.get_training_config()

    assert data_config.max_images_per_class > 0
    assert training_config.batch_size > 0
