import yaml

def load_config(config_path:str) ->dict:
    with open(config_path,'r') as f:
        config=yaml.safe_load(f) 
    #print(config)
    return config


#load_config("config/config.yaml")