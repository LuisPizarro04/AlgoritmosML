import pandas as pd

d = pd.read_csv("datos_finales_csv.csv")  # Importación del archivo csv
datos = pd.DataFrame(d)  # Transformación del archivo csv a un DataFrame

new_dataF = pd.DataFrame()  # Se crea un nuevo DataFrame para obtener una mejor lectura de los datos

new_dataF["edad"] = datos["Edad"]  # SI
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

print(new_dataF.info())
new_dataF.to_csv('prueba_5.csv', index=False)
