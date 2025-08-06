from tools.gpt_get_weights import download_and_load_gpt
WEIGHTS_PATH = './weights/'
MODEL_SIZE = "774M"

settings, params = download_and_load_gpt(MODEL_SIZE, WEIGHTS_PATH)
