import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("datos_finales_csv.csv")  # Importación del archivo csv
datos = pd.DataFrame(d)  # Transformación del archivo csv a un DataFrame
"""
El DataFrame "datos" contiene el archivo original con los datos de la encuesta
# print(datos.info())
"""

"""
Info del dataframe
print(datos.isna())
print(datos.head())
"""
print(datos.info())

print(datos["rango_a_2"])


new_dataF = pd.DataFrame()  # Se crea un nuevo DataFrame para obtener una mejor lectura de los datos

new_dataF["nac"] = datos["nacionalidad"]
new_dataF["edad"] = datos["Edad"]  # SI
new_dataF["sexo"] = datos["sexo"]  # SI
new_dataF["reg"] = datos["region"]
new_dataF["sit_lab"] = datos["sit_Laboral"]  # SI
new_dataF["niv_ing"] = datos["niv_Ingreso"]  # SI
new_dataF["prev"] = datos["fonasa_a_b"].agg(str) + datos["fonasa_c_d"].agg(str) + datos["isapre"].agg(str) + \
                    datos["prais_dipreca_capredena"].agg(str)  # SI
new_dataF["cntr_sld"] = datos["cesfam"].agg(pd.to_numeric).agg(str) + datos["emr"].agg(str) + datos["pr"].agg(str) + \
                        datos["sapu"].agg(str) + datos["cosam"].agg(str) + datos["cecof"].agg(str) + \
                        datos["cgu"].agg(str) + datos["csu"].agg(str) + datos["csr"].agg(str)
new_dataF["motiv_asistencia"] = datos["motiv_asistencia_1"]
new_dataF["forma_solicitar"] = datos["forma_solicitar"]  # SI
new_dataF["hora_ida"] = datos["hora_ida"]
new_dataF["frec_asistencia"] = datos["frec_asistencia"]
new_dataF["hora_ideal"] = datos["rango_a"].agg(str) + datos["rango_b"].agg(str) + datos["rango_c"].agg(str) + \
                          datos["rango_d"].agg(str) + datos["rango_e"].agg(str)  # SI
new_dataF["left_trab_est_1"] = datos["left_trab_est_1"]  # SI
new_dataF["left_trab_est_2"] = datos["left_trab_est_2"]  # SI
new_dataF["dist_hgr_cntr"] = datos["dist_hgr_cntr"]  # NO
new_dataF["hw_transp_cntr"] = datos["hw_transp_cntr"]  # SI
new_dataF["how_time_solici"] = datos["how_time_solici"]  # SI
new_dataF["enf_cron"] = datos["enfer_cro_1"].agg(str) + datos["enfer_cro_2"].agg(str) + \
                        datos["enfer_cro_3"].agg(str) + datos["enfer_cro_4"].agg(str) + \
                        datos["enfer_cro_5"].agg(str) + datos["enfer_cro_6"].agg(str) + \
                        datos["enfer_cro_7"].agg(str) + datos["enfer_cro_8"].agg(str) + \
                        datos["enfer_cro_9"].agg(str) + datos["enfer_cro_10"].agg(str) + \
                        datos["enfer_cro_11"].agg(str) + datos["enfer_cro_12"].agg(str) + \
                        datos["enfer_cro_13"].agg(str) + datos["enfer_cro_14"].agg(str) + \
                        datos["enfer_cro_15"].agg(str) + datos["enfer_cro_16"].agg(str) + \
                        datos["enfer_cro_17"].agg(str)

new_dataF["nece_compa"] = datos["nece_compa"]  # SI
new_dataF["asiste_a_otro"] = datos["asiste_a_otro"]  # SI
new_dataF["hijos"] = datos["hijos"]  # SI
new_dataF["cant_hijos"] = datos["cant_hijos"]  # SI
new_dataF["edad_hj"] = datos["rango_a_2"].agg(str) + datos["rango_b_2"].agg(str) + datos["rango_c_2"].agg(str) + \
                       datos["rango_d_2"].agg(str)  # SI
new_dataF["cuidado_hijos?"] = datos["cuidado_hijos?"]  # NO
new_dataF["apoyo_otra_per"] = datos["apoyo_otra_per"]  # NO
new_dataF["satf_tmp_wait"] = datos["satf_tmp_wait"]  # SI para solicitar
new_dataF["satf_tmp_wait_2"] = datos["satf_tmp_wait_2"]  # SI espera del dia para la cita
new_dataF["percp_1"] = datos["insatisfecho_1"].agg(str) + datos["med_satisfecho_1"].agg(str) + \
                       datos["satisfecho_1"].agg(str)  # SI servicio de entrega de cita
new_dataF["percp_2"] = datos["insatisfecho_2"].agg(str) + datos["med_satisfecho_2"].agg(str) + \
                       datos["satisfecho_2"].agg(str)  # SI métodos de comunicación
new_dataF["percp_3"] = datos["insatisfecho_3"].agg(str) + datos["med_satisfecho_3"].agg(str) + \
                       datos["satisfecho_3"].agg(str)  # SI comprensión de la necesidad del paciente
new_dataF["percp_4"] = datos["insatisfecho_4"].agg(str) + datos["med_satisfecho_4"].agg(str) + \
                       datos["satisfecho_4"].agg(str)  # SI atención médica

new_dataF.to_csv('example.csv', index=False)

print("Grafico 1.....OK")
plt.figure(figsize=(8, 5))
new_dataF["edad"].hist(bins=50, color='#730202', edgecolor='black')
plt.xlabel("Edades")
plt.ylabel("Cantidad de personas")
plt.title("Rangos de edad")
plt.style.use("ggplot")
plt.savefig('grafico_1.png', dpi=300)
plt.show()

print("Grafico 2.....OK")
plt.figure(figsize=(8, 5))
hom = new_dataF[new_dataF["sexo"] == 2].count()[0]
muj = new_dataF[new_dataF["sexo"] == 1].count()[0]
plt.figure(figsize=(8, 5))
labels = ['Hombre ', 'Mujer']
colors = ['#F29F05', '#D97904']
plt.pie([hom, muj], colors=colors, autopct='%.2f %%')
plt.title('Porcentaje de hombres y mujeres')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('grafico_2.png', dpi=300)
plt.show()

print("Grafico 3.....OK")
plt.figure(figsize=(8, 5))
var_1 = new_dataF[new_dataF["sit_lab"] == 1].count()[0]
var_2 = new_dataF[new_dataF["sit_lab"] == 2].count()[0]
var_3 = new_dataF[new_dataF["sit_lab"] == 3].count()[0]
var_4 = new_dataF[new_dataF["sit_lab"] == 4].count()[0]
var_5 = new_dataF[new_dataF["sit_lab"] == 5].count()[0]
var_6 = new_dataF[new_dataF["sit_lab"] == 6].count()[0]
var_7 = new_dataF[new_dataF["sit_lab"] == 7].count()[0]
var_8 = new_dataF[new_dataF["sit_lab"] == 8].count()[0]
var_9 = new_dataF[new_dataF["sit_lab"] == 9].count()[0]
labels = ["Estudiante", "Trabajo Full Time", "Trabajo Part Time", "Trabaja por cuenta propia",
          "Sin trabajo , pero en busca de trabajo", "Sin trabajo  y no busca trabajo", "Jubilado por edad legal",
          "Dueño (a) de casa", "Jubilado por motivos de salud"]
values = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]
bars = plt.barh(labels, values, color='#BF0404')
plt.xlabel("Cantidad de personas")
plt.ylabel("Situación")
plt.title("Situación laboral")
plt.savefig('grafico_3.png', dpi=300)
plt.show()

print("Grafico 4.....OK")
nving_1 = new_dataF[new_dataF["niv_ing"] == 1].count()[0]
nving_2 = new_dataF[new_dataF["niv_ing"] == 2].count()[0]
nving_3 = new_dataF[new_dataF["niv_ing"] == 3].count()[0]
nving_4 = new_dataF[new_dataF["niv_ing"] == 4].count()[0]
nving_5 = new_dataF[new_dataF["niv_ing"] == 5].count()[0]
nving_6 = new_dataF[new_dataF["niv_ing"] == 6].count()[0]
labels = ["0 - 200.000", "200.001  -  400.000", "400.001 - 600.000", "600.001 - 800.000", "800.001 - 1.000.000",
          "Sobre 1.200.001"]
values = [nving_1, nving_2, nving_3, nving_4, nving_5, nving_6]
plt.figure(figsize=(8, 5))
bars2 = plt.barh(labels, values, color='#BF0404')
plt.ylabel("Rango de ingresos")
plt.xlabel("Cantidad de personas")
plt.title("Nivel de ingresos")
plt.savefig('grafico_4.png', dpi=300)
plt.show()

print("Grafico 5.....OK")
prv_1 = new_dataF[new_dataF["prev"] == "1000"].count()[0]
prv_2 = new_dataF[new_dataF["prev"] == "0100"].count()[0]
prv_3 = new_dataF[new_dataF["prev"] == "0010"].count()[0]
prv_4 = new_dataF[new_dataF["prev"] == "0001"].count()[0]
prv_5 = new_dataF[new_dataF["prev"] == "1001"].count()[0]
prv_6 = new_dataF[new_dataF["prev"] == "0101"].count()[0]
prv_7 = new_dataF[new_dataF["prev"] == "0011"].count()[0]
labels = ['Fonasa (A o B)', 'Fonasa (C o D)', 'Isapre', 'Prais', 'Prais/Fns A-B', 'Prais/Fns C-D', 'Prais/Isapre']
values = [prv_1, prv_2, prv_3, prv_4, prv_5, prv_6, prv_7]
plt.figure(figsize=(8, 5))
bars3 = plt.barh(labels, values, color='#BF0404')
plt.xlabel("Cantidad de personas")
plt.ylabel("Previsiones")
plt.title("Tipo de Previsión")
plt.style.use("ggplot")
plt.savefig('grafico_5.png', dpi=300)
plt.show()

print("Grafico 6.....OK")
forma_1 = new_dataF[new_dataF["forma_solicitar"] == 1].count()[0]
forma_2 = new_dataF[new_dataF["forma_solicitar"] == 2].count()[0]
forma_3 = new_dataF[new_dataF["forma_solicitar"] == 3].count()[0]
plt.figure(figsize=(8, 5))
labels = ['Telefonica ', 'Presencial', 'Vía Web']
colors = ['#730202', '#BF0404', '#D97904']
plt.pie([forma_1, forma_2, forma_3], colors=colors, autopct='%.2f %%')
plt.title('Manera en que solicitan una hora médica')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('grafico_6.png', dpi=300)
plt.show()

print("Grafico 7.....OK")
hdl_1 = new_dataF[new_dataF["hora_ideal"] == "10000"].count()[0]
hdl_2 = new_dataF[new_dataF["hora_ideal"] == "01000"].count()[0]
hdl_3 = new_dataF[new_dataF["hora_ideal"] == "00100"].count()[0]
hdl_4 = new_dataF[new_dataF["hora_ideal"] == "00010"].count()[0]
hdl_5 = new_dataF[new_dataF["hora_ideal"] == "00001"].count()[0]
labels = ['8:30  a 10:00 AM', '11:00 AM a 12:00 PM', '13:00 a 15:00 PM', '16:00 a 18:00 PM', '19:00 a 20:00 PM']
values = [hdl_1, hdl_2, hdl_3, hdl_4, hdl_5]
plt.figure(figsize=(8, 5))
bars4 = plt.barh(labels, values, color='#BF0404')
plt.ylabel("Horas")
plt.xlabel("Cantidad de personas")
plt.title("Hora ideal para una cita médica")
plt.style.use("ggplot")
plt.savefig('grafico_7.png', dpi=300)
plt.show()

print("Grafico 8.....OK")
left_1_1 = new_dataF[new_dataF["left_trab_est_1"] == 1].count()[0]
left_1_2 = new_dataF[new_dataF["left_trab_est_1"] == 2].count()[0]
left_1_3 = new_dataF[new_dataF["left_trab_est_1"] == 3].count()[0]
labels = ["Si", "No", "A veces"]
values = [left_1_1, left_1_2, left_1_3]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color='#BF0404')
plt.ylabel("Cantidad de personas")
plt.xlabel("Respuestas")
plt.title("Cantidad de personas que deben dejar de trabajar o \n estudiar para asistir a una cita médica")
plt.savefig('grafico_8.png', dpi=300)
plt.show()

print("Grafico 9.....OK")
left_2_1 = new_dataF[new_dataF["left_trab_est_2"] == 1].count()[0]
left_2_2 = new_dataF[new_dataF["left_trab_est_2"] == 2].count()[0]
left_2_3 = new_dataF[new_dataF["left_trab_est_2"] == 3].count()[0]
labels = ["Si", "No", "A veces"]
values = [left_2_1, left_2_2, left_2_3]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color='#BF0404')
plt.ylabel("Cantidad de personas")
plt.xlabel("Respuestas")
plt.title("Cantidad de personas que deben dejar de trabajar o  \n estudiar para acompañar a alguien a una cita médica")
plt.savefig('grafico_9.png', dpi=300)
plt.show()

print("Grafico 10.....OK")
transp_1 = new_dataF[new_dataF["hw_transp_cntr"] == 1].count()[0]
transp_2 = new_dataF[new_dataF["hw_transp_cntr"] == 2].count()[0]
transp_3 = new_dataF[new_dataF["hw_transp_cntr"] == 3].count()[0]
labels = ["Transporte Público", "Transporte Privado", "Caminando"]
values = [transp_1, transp_2, transp_3]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color='#BF0404')
plt.ylabel("Cantidad de personas")
plt.xlabel("Respuestas")
plt.title("Manera en que las personas se movilizan \n al centro de salud al que asisten regularmente")
plt.savefig('grafico_10.png', dpi=300)
plt.show()

print("Grafico 11.....OK")
forma_1 = new_dataF[new_dataF["how_time_solici"] == 1].count()[0]
forma_2 = new_dataF[new_dataF["how_time_solici"] == 2].count()[0]
forma_3 = new_dataF[new_dataF["how_time_solici"] == 3].count()[0]
forma_4 = new_dataF[new_dataF["how_time_solici"] == 4].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Menos de una hora ', '1 a 2 horas', '2 a 3 horas', 'Más de 3 horas']
colors = ['#F29F05', '#D97904', '#BF0404', '#730202']
plt.pie([forma_1, forma_2, forma_3, forma_4], colors=colors, autopct='%.2f %%')
plt.title('Tiempo que demoran las personas en solicitar \n una cita medica en el centro de salud al que asisten')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper right')
plt.savefig('grafico_11.png', dpi=300)
plt.show()

print("Grafico 12.....OK")
si_nec = new_dataF[new_dataF["nece_compa"] == 1].count()[0]
no_nec = new_dataF[new_dataF["nece_compa"] == 2].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Si necesita ayuda ', 'No necesita ayuda']
colors = ['#F29F05', '#D97904']
plt.pie([si_nec, no_nec], colors=colors, autopct='%.2f %%')
plt.title('¿Porcentaje de personas que requieren ir acompañados al centro de salud? ')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper right')
plt.savefig('grafico_12.png', dpi=300)
plt.show()

print("Grafico 13.....OK")
si_ayuda = new_dataF[new_dataF["asiste_a_otro"] == 1].count()[0]
no_ayuda = new_dataF[new_dataF["asiste_a_otro"] == 2].count()[0]
plt.figure(figsize=(8, 5))
labels = ['Si ayuda a otros', 'No ayuda a otros']
colors = ['#D97904', '#F29F05']
plt.pie([si_ayuda, no_ayuda], colors=colors, autopct='%.2f %%')
plt.title('¿Porcentaje de personas que acompañan a otros al centro de salud? ')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('grafico_13.png', dpi=300)
plt.show()

print("Grafico 14.....OK")
si_hj = new_dataF[new_dataF["hijos"] == 1].count()[0]
no_hj = new_dataF[new_dataF["hijos"] == 2].count()[0]
plt.figure(figsize=(8, 5))
labels = ['Si tiene hijos', 'No tiene hijos']
colors = ['#D97904', '#F29F05']
plt.pie([si_hj, no_hj], colors=colors, autopct='%.2f %%')
plt.title('Porcentaje de personas que tienen hijos')
plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.savefig('grafico_14.png', dpi=300)
plt.show()

print("Grafico 15.....OK")
hj_1 = new_dataF[new_dataF["cant_hijos"] == 1].count()[0]
hj_2 = new_dataF[new_dataF["cant_hijos"] == 2].count()[0]
hj_3 = new_dataF[new_dataF["cant_hijos"] == 3].count()[0]
hj_4 = new_dataF[new_dataF["cant_hijos"] == 4].count()[0]
hj_5 = new_dataF[new_dataF["cant_hijos"] == 5].count()[0]
labels = ["0", "1", "2", "3", "+3"]
values = [hj_1, hj_2, hj_3, hj_4, hj_5]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color='#BF0404')
plt.ylabel("Cantidad de respuestas")
plt.xlabel("Cantidad de hijos")
plt.title("Cantidad de hijos de las personas encuestadas")
plt.savefig('grafico_15.png', dpi=300)
plt.show()

print("Grafico 16.....OK")
stw_1_1 = new_dataF[new_dataF["satf_tmp_wait"] == 1].count()[0]
stw_1_2 = new_dataF[new_dataF["satf_tmp_wait"] == 2].count()[0]
stw_1_3 = new_dataF[new_dataF["satf_tmp_wait"] == 3].count()[0]
stw_1_4 = new_dataF[new_dataF["satf_tmp_wait"] == 4].count()[0]
stw_1_5 = new_dataF[new_dataF["satf_tmp_wait"] == 5].count()[0]
labels = ["De alguna manera insatisfecho", "Insatisfecho", "Neutral", "Algo satisfecho", "Muy satisfecho"]
values = [stw_1_1, stw_1_2, stw_1_3, stw_1_4, stw_1_5]
plt.figure(figsize=(8, 5))
plt.barh(labels, values, color='#BF0404')
plt.ylabel("Nivel de satisfacción")
plt.xlabel("Cantidad de respuestas")
plt.title('Nivel de satisfacción de: "Los tiempos de espera \n para agendar citas médicas"')
plt.savefig('grafico_16.png', dpi=300)
plt.show()

print("Grafico 17.....OK")
stw_2_1 = new_dataF[new_dataF["satf_tmp_wait_2"] == 1].count()[0]
stw_2_2 = new_dataF[new_dataF["satf_tmp_wait_2"] == 2].count()[0]
stw_2_3 = new_dataF[new_dataF["satf_tmp_wait_2"] == 3].count()[0]
stw_2_4 = new_dataF[new_dataF["satf_tmp_wait_2"] == 4].count()[0]
stw_2_5 = new_dataF[new_dataF["satf_tmp_wait_2"] == 5].count()[0]
labels = ["De alguna manera insatisfecho", "Insatisfecho", "Neutral", "Algo satisfecho", "Muy satisfecho"]
values = [stw_2_1, stw_2_2, stw_2_3, stw_2_4, stw_2_5]
plt.figure(figsize=(8, 5))
plt.barh(labels, values, color='#BF0404')
plt.ylabel("Nivel de satisfacción")
plt.xlabel("Cantidad de respuestas")
plt.title('Nivel de satisfacción de: "Los tiempos de espera \n el día de la cita"')
plt.savefig('grafico_17.png', dpi=300)
plt.show()

print("Grafico 18.....OK")
pcp_1_1 = new_dataF[new_dataF["percp_1"] == "100"].count()[0]
pcp_1_2 = new_dataF[new_dataF["percp_1"] == "010"].count()[0]
pcp_1_3 = new_dataF[new_dataF["percp_1"] == "001"].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Insatisfecho/a', 'Medianamente satisfecho/a', 'Satisfecho/a']
colors = ['#F29F05', '#D97904', '#BF0404']
plt.pie([pcp_1_1, pcp_1_2, pcp_1_3], colors=colors, autopct='%.2f %%')
plt.title('Nivel de satisfacción de: "El servicio de entrega de citas"')
plt.legend(labels, loc='upper right')
plt.savefig('grafico_18.png', dpi=300)
plt.show()

print("Grafico 19.....OK")
pcp_2_1 = new_dataF[new_dataF["percp_2"] == "100"].count()[0]
pcp_2_2 = new_dataF[new_dataF["percp_2"] == "010"].count()[0]
pcp_2_3 = new_dataF[new_dataF["percp_2"] == "001"].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Insatisfecho/a', 'Medianamente satisfecho/a', 'Satisfecho/a']
colors = ['#F29F05', '#D97904', '#BF0404']
plt.pie([pcp_2_1, pcp_2_2, pcp_2_3], colors=colors, autopct='%.2f %%')
plt.title('Nivel de satisfacción de: \n "Los métodos de comunicación para agendar horas"')
plt.legend(labels, loc='upper right')
plt.savefig('grafico_19.png', dpi=300)
plt.show()

print("Grafico 20.....OK")
pcp_3_1 = new_dataF[new_dataF["percp_3"] == "100"].count()[0]
pcp_3_2 = new_dataF[new_dataF["percp_3"] == "010"].count()[0]
pcp_3_3 = new_dataF[new_dataF["percp_3"] == "001"].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Insatisfecho/a', 'Medianamente satisfecho/a', 'Satisfecho/a']
colors = ['#F29F05', '#D97904', '#BF0404']
plt.pie([pcp_3_1, pcp_3_2, pcp_3_3], colors=colors, autopct='%.2f %%')
plt.title('Nivel de satisfacción de: \n "La comprensión de las necesidades del paciente"')
plt.legend(labels, loc='upper right')
plt.savefig('grafico_20.png', dpi=300)
plt.show()

print("Grafico 21.....OK")
pcp_4_1 = new_dataF[new_dataF["percp_4"] == "100"].count()[0]
pcp_4_2 = new_dataF[new_dataF["percp_4"] == "010"].count()[0]
pcp_4_3 = new_dataF[new_dataF["percp_4"] == "001"].count()[0]
plt.figure(figsize=(10, 6))
labels = ['Insatisfecho/a', 'Medianamente satisfecho/a', 'Satisfecho/a']
colors = ['#F29F05', '#D97904', '#BF0404']
plt.pie([pcp_4_1, pcp_4_2, pcp_4_3], colors=colors, autopct='%.2f %%')
plt.title('Nivel de satisfacción de: "La Atención médica"')
plt.legend(labels, loc='upper right')
plt.savefig('grafico_21.png', dpi=300)
plt.show()


