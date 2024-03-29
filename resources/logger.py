import os
import json

class CSVLogger():
    def __init__(self, log_dir:str) -> None:
        self.log_dir = log_dir

        # make sure log directory exists:
        os.makedirs(self.log_dir, exist_ok=True)

        self.files = []
        self.log_path = None
        self.meta_path = None

    def create_file(self, file_name:str, header:list=[]):
        if not file_name.endswith(".csv"):
            file_name = file_name + ".csv"
        
        self.files.append(file_name)
        self.log_path = os.path.join(self.log_dir, file_name)

        header_str = ",".join([str(h) for h in header])

        with open(self.log_path, "w") as f:
            f.write(header_str+"\n")

    def push(self, values:list):
        values_str = ",".join([str(v) for v in values])
        with open(self.log_path, "a") as f:
            f.write(values_str+"\n")

        
    def log_meta_data(self, meta_info:dict):
        meta_file_name = self.files[-1][:-4] # get filename without .csv
        meta_file_name += "_meta.json"
        self.meta_path = os.path.join(self.log_dir, meta_file_name)

        meta_info_str = json.dumps(meta_info)

        with open(self.meta_path, "w") as f:
            f.write(meta_info_str)
            # for info in meta_info:
            #     f.write(info)
            #     f.write("\n")
