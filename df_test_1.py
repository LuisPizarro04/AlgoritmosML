import pandas as pd
import numpy as np

d = pd.read_csv("datos_finales_csv.csv")  # Importación del archivo csv
datos = pd.DataFrame(d)  # Transformación del archivo csv a un DataFrame

new_dataF = pd.DataFrame()  # Se crea un nuevo DataFrame para obtener una mejor lectura de los datos

new_dataF["nac"] = datos["nacionalidad"]
new_dataF["edad"] = datos["Edad"]  # SI
new_dataF["sexo"] = datos["sexo"]  # SI
new_dataF["reg"] = datos["region"]
new_dataF["sit_lab"] = datos["sit_Laboral"]  # SI
new_dataF["niv_ing"] = datos["niv_Ingreso"]  # SI
new_dataF["prev"] = datos["fonasa_a_b"].agg(str) + datos["fonasa_c_d"].agg(str) + datos["isapre"].agg(str) + \
                    datos["prais_dipreca_capredena"].agg(str)  # SI


def prev_fix(pr):
    if pr == "1000":
        return 1
    elif pr == "0100":
        return 2
    elif pr == "0010":
        return 3
    elif pr == "0001":
        return 4
    elif pr == "1001":
        return 5
    elif pr == "0101":
        return 6
    elif pr == "0011":
        return 7
    else:
        return 0


new_dataF["prev"] = new_dataF.apply(lambda x: prev_fix(x["prev"]), axis=1)
new_dataF["motiv_asistencia"] = datos["motiv_asistencia_1"]
new_dataF["forma_solicitar"] = datos["forma_solicitar"]  # SI
new_dataF["hora_ida"] = datos["hora_ida"]
new_dataF["frec_asistencia"] = datos["frec_asistencia"]
new_dataF["hora_ideal"] = datos["rango_a"].agg(str) + datos["rango_b"].agg(str) + datos["rango_c"].agg(str) + \
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
    elif hif == "00001":
        return 5
    else:
        return 0


new_dataF["hora_ideal"] = new_dataF.apply(lambda z: hor_ideal_fix(z["hora_ideal"]), axis=1)
new_dataF["left_trab_est_1"] = datos["left_trab_est_1"]  # SI
new_dataF["left_trab_est_2"] = datos["left_trab_est_2"]  # SI
new_dataF["dist_hgr_cntr"] = datos["dist_hgr_cntr"]  # NO
new_dataF["hw_transp_cntr"] = datos["hw_transp_cntr"]  # SI
new_dataF["how_time_solici"] = datos["how_time_solici"]  # SI
new_dataF["enf_cron"] = datos["tiene_enf_cronic"]
new_dataF["nece_compa"] = datos["nece_compa"]  # SI
new_dataF["asiste_a_otro"] = datos["asiste_a_otro"]  # SI
new_dataF["hijos"] = datos["hijos"]  # SI
new_dataF["cant_hijos"] = datos["cant_hijos"]  # SI
conditions_1 = [(datos['rango_a_2'] == 1) | (datos['rango_b_2'] == 1) | (datos['rango_c_2'] == 1),
                (datos['rango_a_2'] == 0) & (datos['rango_b_2'] == 0) & (datos['rango_c_2'] == 0) &
                (datos['rango_d_2'] == 1), (datos['rango_a_2'] == 0) & (datos['rango_b_2'] == 0) &
                (datos['rango_c_2'] == 0) & (datos['rango_d_2'] == 0)]
values_1 = [1, 0, 0]
new_dataF["dep_son"] = np.select(conditions_1, values_1)
new_dataF["cuidado_hijos?"] = datos["cuidado_hijos?"]  # NO
new_dataF["apoyo_otra_per"] = datos["apoyo_otra_per"]  # NO
new_dataF["satf_tmp_wait"] = datos["satf_tmp_wait"]  # SI para solicitar
new_dataF["satf_tmp_wait_2"] = datos["satf_tmp_wait_2"]  # SI espera del dia para la cita
conditions_a = [(datos["insatisfecho_1"] == 1) & (datos["med_satisfecho_1"] == 0) & (datos["satisfecho_1"] == 0),
                (datos["insatisfecho_1"] == 0) & (datos["med_satisfecho_1"] == 1) & (datos["satisfecho_1"] == 0),
                (datos["insatisfecho_1"] == 0) & (datos["med_satisfecho_1"] == 0) & (datos["satisfecho_1"] == 1)]
values_a = [1, 2, 3]
new_dataF["percp_1"] = np.select(conditions_a, values_a)
conditions_b = [(datos["insatisfecho_2"] == 1) & (datos["med_satisfecho_2"] == 0) & (datos["satisfecho_2"] == 0),
                (datos["insatisfecho_2"] == 0) & (datos["med_satisfecho_2"] == 1) & (datos["satisfecho_2"] == 0),
                (datos["insatisfecho_2"] == 0) & (datos["med_satisfecho_2"] == 0) & (datos["satisfecho_2"] == 1)]
values_b = [1, 2, 3]
new_dataF["percp_2"] = np.select(conditions_b, values_b)
conditions_c = [(datos["insatisfecho_3"] == 1) & (datos["med_satisfecho_3"] == 0) & (datos["satisfecho_3"] == 0),
                (datos["insatisfecho_3"] == 0) & (datos["med_satisfecho_3"] == 1) & (datos["satisfecho_3"] == 0),
                (datos["insatisfecho_3"] == 0) & (datos["med_satisfecho_3"] == 0) & (datos["satisfecho_3"] == 1)]
values_c = [1, 2, 3]
new_dataF["percp_3"] = np.select(conditions_c, values_c)
conditions_d = [(datos["insatisfecho_4"] == 1) & (datos["med_satisfecho_4"] == 0) & (datos["satisfecho_4"] == 0),
                (datos["insatisfecho_4"] == 0) & (datos["med_satisfecho_4"] == 1) & (datos["satisfecho_4"] == 0),
                (datos["insatisfecho_4"] == 0) & (datos["med_satisfecho_4"] == 0) & (datos["satisfecho_4"] == 1)]
values_d = [1, 2, 3]
new_dataF["percp_4"] = np.select(conditions_d, values_d)
print(new_dataF.info())
new_dataF.to_csv('prueba_1.csv', index=False)
