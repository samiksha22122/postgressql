from core.utils.read_config import config_manager

def get_database_client(database_type: str):
    config = config_manager.load_config("config.yaml")  # make sure this function is implemented
    logging.info("Loaded database configs inside utils")

    try:
        db_config = config["database_configs"][database_type]
        module_name = db_config["module"]
        class_name = db_config["class"]
    except KeyError:
        raise ValueError(f"Unsupported database type: {database_type}")

    module = import_module(module_name)
    return getattr(module, class_name)()
