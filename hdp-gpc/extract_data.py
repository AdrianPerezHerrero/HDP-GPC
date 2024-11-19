import os
import numpy as np
homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("D:/Programs/Workspaces/spyder-workspace/HDP-GPC")
else:
    try:
        os.chdir(homedir + "/Documents/HDP-GPC")
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

from get_data import get_data, take_standard_labels
list_rec=["100", "101", "102", "103", "104", "105", "106", "107",
          "108", "109", "111", "112", "113", "114", "115", "116",
          "117", "118", "119", "121", "122", "123", "124", "200",
          "201", "202", "203", "205", "207", "208", "209", "210",
          "212", "213", "214", "215", "217", "219", "220", "221",
          "222", "223", "228", "230", "231", "232", "233", "234"]
for record in list_rec:

    samples = [60, 150]
    M = 2
    #samples = [0, -1]
    data, labels = get_data(database="filtered", record=record, deriv=None, test=False,
                            scale_data=True, scale_type="mean", d2_data=False, samples=samples, ann='atr')
    dat_ = data
    data, data_2d, labels = take_standard_labels(data, labels, filter=labels)
    np.save(cwd + "/data/mitbih/"+record+".npy", data_2d)
    np.save(cwd + "/data/mitbih/"+record+"_labels.npy", labels)
print("END")