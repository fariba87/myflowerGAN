import json, numpy as np


class ConfigReader(object):
    def __init__(self):
        with open("./ConFig/config.json", "r") as f:
            cfg = json.load(f)
        self.modelName = cfg["modelName"]
        self.SanityCheck = cfg["SanityCheck"] == "True"
        self.InChannel = np.int0(cfg["InChannel"])
        self.batchSize = np.int0(cfg["batchSize"])
        self.TotalEpoch = cfg["TotalEpoch"]
        self.img_height = cfg["img_height"]
        self.img_width = cfg["img_height"]
        self.projectionSize = [4, 4, 512] #cfg["projectionSize"]
        self.Shuffle = cfg["Shuffle"]
        self.Adam_lr = cfg["Adamlr"]
        self.Adam_gradientdecay = cfg["Adamgradientdecay"]
        self.Adam_square_gd = cfg["Adamsquaregd"]
        self.stride_G = cfg["strideG"]
        self.stride_D = cfg["strideD"]
        self.noise_dim = cfg["codingsize"]
        self.padding = cfg["padding"]
        self.kernel_size = cfg["kernelsize"]
        self.dropout_rate = cfg["dropoutrate"]
        self.num_filter = cfg["numfilter"]
        self.scale_filter_G = cfg["scalefilterG"]
        self.scale_filter_D = cfg["scalefilterD"]
        self.leakyreluscale = cfg["leakyreluscale"]
        self.MainPath = ""
