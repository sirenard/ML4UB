import math

import ecole
import egret.models.unit_commitment
import numpy as np
import pyomo.environ as pyo
import pyscipopt
from pyscipopt import *
import egret.data.model_data as md

from egret.data.model_data import ModelData

"""
Generate unit commitment instances from a Ucig file that contains 1 problems with n days.
The generator split the problem in problems with limited number of days.
Thus the instances share the same generators while the demand of each problems changes
"""
class UcigGenerator:
    def __init__(self, file_path, days=1, start_at_instance=0):
        """
        Take as input a file generated with ucig and only thermal units
        The generator generates 1 UC instance for each day of the input file (so keeps the same config)
        :param file_path:
        :param horizon:
        """
        self.file = open(file_path, "r")
        self.lines = iter(self.file.readlines())
        self.model_data = None
        self.days = days
        self.loads = []
        self.count=0

        for _ in range(start_at_instance):
            self.generateNext()

    def __del__(self):
        self.file.close()



    def get_next_modeldata(self):
        data = {}

        single_values = [
            "ProblemNum",
            "HorizonLen",
            "NumThermal",
            "NumHydro",
            "NumCascade",
            "LoadCurve",
            "MinSystemCapacity",
            "MaxSystemCapacity",
            "MaxThermalCapacity",
        ]
        while True:
            line = next(self.lines)
            line = line.strip().split()
            key = line[0]

            if key == "HydroCascadeSection":
                break

            if key in single_values and len(line)>1:
                data[key] = float(line[1])

            elif key == "Loads":
                loads = []
                n_days = int(line[1])
                n_splits = int(line[2])
                for i in range(n_days):
                    values = list(map(float, next(self.lines).strip().split()))
                    loads.append(values)
                data[key] = loads

            elif key == "SpinningReserve":
                values = list(map(float, next(self.lines).strip().split()))
                data[key] = values

            elif key == "ThermalSection":
                thermal_units = []
                for i in range(int(data["NumThermal"])):
                    values = list(map(float, next(self.lines).strip().split()))
                    thermal_units.append(values)

                data[key] = thermal_units

        model_data = md.ModelData.empty_model_data_dict()
        system = model_data["system"]
        elements = model_data["elements"]

        time_periods = list(range(n_splits*self.days))

        #system["reserve_requirement"] = {"data_type": "time_series", "values": dict(zip(range(1,1+len(data["SpinningReserve"])),data["SpinningReserve"]))}
        system["reserve_requirement"] = {"data_type": "time_series", "values": data["SpinningReserve"]*self.days}
        system["time_keys"] = time_periods
        system["baseMVA"] = 1.
        system["time_period_length_minutes"] = 60
        system["load_mismatch_cost"] = 10000
        system["reserve_shortfall_cost"] = 1000
        system["reference_bus"] = "copperplate"
        system["reference_bus_angle"] = 0.

        elements["bus"] = {"copperplate": dict()}

        elements["load"] = {"demand": {"bus": "copperplate",
                                       "in_service": True,
                                       "p_load": {"data_type": "time_series", "values": {}
                                            }
                                       }
                            }

        elements["branch"] = dict()
        elements["zone"] = dict()

        generators = dict()


        for i,d in enumerate(data["ThermalSection"]):
            md_g_dict = {"generator_type": "thermal", "bus": "copperplate", "fuel": "G", "in_service": True}
            md_g_dict["p_cost"] = { 'data_type': 'cost_curve', 'cost_curve_type': 'polynomial', 'values': { 0: d[3], 1: d[2], 2: d[1]} }
            md_g_dict["p_min"] = d[4]
            md_g_dict["p_max"] = d[5]
            md_g_dict["initial_status"] = d[6]
            md_g_dict["min_up_time"] = d[8]
            md_g_dict["min_down_time"] = d[9]
            md_g_dict["ramp_up_60min"] = (d[5]-d[4])/4
            md_g_dict["ramp_down_60min"] = (d[5]-d[4])/4
            md_g_dict["initial_p_output"] = d[4]
            md_g_dict["initial_p_output"] = d[4]

            coolAndFuelCost, hotAndFuelCost, tau, tauMax, fixedCost, SUCC = d[10:16]
            md_g_dict["startup_cost"] = [(downTime, coolAndFuelCost * (1 - math.e**(- downTime / tau)) + fixedCost) for downTime in range(0, 1)]

            generators[f"GEN{int(d[0])}"] = md_g_dict

        elements["generator"] = generators


        self.model_data = model_data
        self.loads = data["Loads"]


    def __iter__(self):
        return self

    def generateNext(self):
        if self.count >= len(self.loads)*self.days:
            self.get_next_modeldata()
            self.count = 0

        load = []
        for d in range(self.days):
            load += self.loads[self.count+d]

        self.model_data["elements"]["load"]["demand"]["p_load"]["values"] = load
        self.count += self.days


    def __next__(self):
        self.generateNext()
        m = ModelData(self.model_data)

        pyomo_model = egret.models.unit_commitment.create_CA_unit_commitment_model(m)
        pyomo_model.write("prob.lp")
        model_ecole = ecole.scip.Model.from_file("prob.lp")
        return model_ecole


if __name__ == "__main__":
    g = UcigGenerator("/home/simon/Téléchargements/ucig/instances")
    for model in g:
        model: pyo.ConcreteModel
        model.write("prob.lp")
        pass
