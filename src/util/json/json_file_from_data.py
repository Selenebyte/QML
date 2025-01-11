import os

import json


def save_model_data_to_json_file(model_data, is_noise_model_data=False):
    """Dump the model data to a file.

    Dump the json format of the model data to a file at location: "data/[X]/data-[name].json".
    [x] being determined to be "noise_data" or "non_noise_data" whether "is_noise_model_data" is True or False.
    [name] is the first key of the json format of the model data.

    Args:
                    model_data: JSON format of the model data.
        is_noise_model_data: Boolean of whether the model data is considered noise data.

    returns:
          N/A.

    Raises:
                    FileNotFoundError: An error occured when opening the file. File may not exist.
    """
    if is_noise_model_data:
        filename = f"data/noise_models/data-{list(model_data.keys())[0]}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(model_data, file, ensure_ascii=False, indent=2)
    else:
        filename = f"data/non_noise_models/data-{list(model_data.keys())[0]}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(model_data, file, ensure_ascii=False, indent=2)
