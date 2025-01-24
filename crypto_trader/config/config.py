import os
import yaml


class Config:
    """
    Config class for managing application configuration.

    This class loads configuration from a YAML file.
    It implements the Singleton pattern to ensure only one instance of the
    configuration is created.

    Attributes:
        _instance (Config): The singleton instance of the Config class.
        _config (dict): The dictionary containing configuration data.
    """

    def __init__(self):
        """
        Loads configuration from a YAML file.

        This method reads configuration from a YAML file. Environment variables
        can override the values in the YAML file if needed.
        """
        # Path to the configuration files
        conf_paths = {
            "params": "crypto_trader/config/parameters.yaml",
            "credentials": "crypto_trader/config/credentials.yaml",
        }
        for attr_name, path in conf_paths.items():
            # Load configuration from the YAML file
            with open(path, "r") as config_file:
                setattr(self, attr_name, yaml.safe_load(config_file))

        # Convert ticker_exceptions to a dictionary of tuples
        ticker_exceptions = self.params["tickers"]["exceptions"]
        exceptions_dict = {tuple(item[0]): tuple(item[1]) for item in ticker_exceptions}
        self.params["tickers"]["exceptions"] = exceptions_dict
        self.params["tickers"]["reverse_map"] = dict(
            zip(exceptions_dict.values(), exceptions_dict.keys())
        )


if __name__ == "__main__":
    # Create an instance of the Config class
    config = Config()

    # Access configuration values
    print(f"Fiats: {config.params["tickers"]["fiats"]}")
    print(f"Exceptions: {config.params["tickers"]["exceptions"]}")
