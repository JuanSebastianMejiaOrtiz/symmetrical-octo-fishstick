import pandas as pd


def loadSensorData(path):
    return pd.read_csv(path)


def getSensorData(path):
    data = loadSensorData(path)

# TODO: Check if this is good
    temperature = data["temperature"].values
    voltage = data["voltage"].values
    return temperature, voltage
