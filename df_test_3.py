import pandas as pd
import numpy as np

d = pd.read_csv("datos_finales_csv.csv")  # Importación del archivo csv
datos = pd.DataFrame(d)  # Transformación del archivo csv a un DataFrame
data_ML = pd.DataFrame()

data_ML["edad"] = datos["Edad"]  # SI
data_ML["hijos"] = datos["hijos"]  # SI

conditions_1 = [(datos['rango_a_2'] == 1) | (datos['rango_b_2'] == 1) | (datos['rango_c_2'] == 1),
                (datos['rango_a_2'] == 0) & (datos['rango_b_2'] == 0) & (datos['rango_c_2'] == 0) &
                (datos['rango_d_2'] == 1), (datos['rango_a_2'] == 0) & (datos['rango_b_2'] == 0) &
                (datos['rango_c_2'] == 0) & (datos['rango_d_2'] == 0)]

values_1 = [1, 0, 0]
data_ML["dep_son"] = np.select(conditions_1, values_1)
data_ML["enf_cron"] = datos["tiene_enf_cronic"]
data_ML["nece_compa"] = datos["nece_compa"]  # SI
data_ML["hw_transp_cntr"] = datos["hw_transp_cntr"]  # SI
data_ML["dist"] = datos["dist_hgr_cntr"]

conditions_2 = [(datos["sit_Laboral"] == 1), (datos["sit_Laboral"] == 2), (datos["sit_Laboral"] == 3)]
values_2 = [1, 2, 3]

data_ML["sit_lab"] = np.select(conditions_2, values_2)
data_ML["left_trab_est_1"] = datos["left_trab_est_1"]  # SI
data_ML["hora_ideal"] = datos["rango_a"].agg(str) + datos["rango_b"].agg(str) + datos["rango_c"].agg(str) + \
                        datos["rango_d"].agg(str) + datos["rango_e"].agg(str)  # SI


def hor_ideal_fix(hif):
    if hif == "10000":
        return 1
    elif hif == "01000":
        return 2
    elif hif == "00100":
        return 3
    elif hif == "00010":
        return 4
    else:
        return 5


data_ML["hora_ideal"] = data_ML.apply(lambda z: hor_ideal_fix(z["hora_ideal"]), axis=1)

print(data_ML.info())
print(data_ML.head())
data_ML.to_csv('prueba_3.csv', index=False)
