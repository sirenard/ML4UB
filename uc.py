import math

import ecole
import egret.models.unit_commitment
import numpy as np
import pyomo.environ as pyo
import pyscipopt
from pyscipopt import *
import egret.data.model_data as md

from egret.data.model_data import ModelData

class UcigGenerator:
    def __init__(self, file_path, days=1, start_at_instance=0):
        """
        Take as input a file generated with ucig and only thermal units
        The generator generates 1 UC instance for each day of the input file (so keeps the same config)
        :param file_path:
        :param horizon:
        """
        data = {}
        with open(file_path, "r") as f:
            lines = iter(f.readlines())

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
            for line in lines:
                line = line.strip().split()
                key = line[0]

                if key in single_values and len(line)>1:
                    data[key] = float(line[1])

                elif key == "Loads":
                    loads = []
                    n_days = int(line[1])
                    n_splits = int(line[2])
                    for i in range(n_days):
                        values = list(map(float, next(lines).strip().split()))
                        loads.append(values)
                    data[key] = loads

                elif key == "SpinningReserve":
                    values = list(map(float, next(lines).strip().split()))
                    data[key] = values

                elif key == "ThermalSection":
                    thermal_units = []
                    for i in range(int(data["NumThermal"])):
                        values = list(map(float, next(lines).strip().split()))
                        thermal_units.append(values)

                    data[key] = thermal_units

        model_data = md.ModelData.empty_model_data_dict()
        system = model_data["system"]
        elements = model_data["elements"]

        time_periods = list(range(n_splits*days))

        #system["reserve_requirement"] = {"data_type": "time_series", "values": dict(zip(range(1,1+len(data["SpinningReserve"])),data["SpinningReserve"]))}
        system["reserve_requirement"] = {"data_type": "time_series", "values": data["SpinningReserve"]*days}
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
        self.days = days
        self.count = start_at_instance

    def __iter__(self):
        return self

    def __next__(self):
        if self.count > len(self.loads):
            raise StopIteration

        load = []
        for d in range(self.days):
            load += self.loads[self.count+d]

        #load_dic: dict = self.model_data["elements"]["load"]["demand"]["p_load"]["values"]
        #load_dic.clear()
        #for i,val in enumerate(load):
        #    load_dic[i+1]=val

        self.model_data["elements"]["load"]["demand"]["p_load"]["values"] = load

        m = ModelData(self.model_data)
        self.count += self.days

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
