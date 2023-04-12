import os

def create_evaldir(path, eval_name):
    result = os.path.join(path, eval_name)
    os.makedirs(result, exist_ok = True)

    return result
