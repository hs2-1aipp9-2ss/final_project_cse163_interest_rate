import os
import pandas as pd


def japan_process_file(dir: str) -> pd.DataFrame:
    filenames = os.listdir(dir)
    fname_pd_dict: dict[str, pd.DataFrame] = {}
    for filename in filenames:
        pathname = os.path.join(dir, filename)
        if os.path.isfile(pathname):
            if os.path.splitext(pathname)[1] == ".csv":
                fname_pd_dict[filename] = pd.read_csv(pathname)
    
    return fname_pd_dict


def main():
    directory = "./Data/Japan/"
    japan = japan_process_file(directory)
    print(japan)

if __name__ == "__main__":
    main() 