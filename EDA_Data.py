import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
d = pd.read_csv("prueba_4.csv")
new_dataF = pd.DataFrame(d)
# Nivel de selecciones de horarios.

plt.style.use("ggplot")

grupo = new_dataF.groupby('hora_ideal').size()
print(grupo)
labels = ['8:30  a 10:00 AM', '11:00 AM a 12:00 PM', '13:00 a 15:00 PM', '16:00 a 18:00 PM', '19:00 a 20:00 PM']
grupo.plot(kind='bar', title='Respuestas por "Hora ideal" \n para una cita médica')
plt.xticks(rotation=0)
plt.savefig('var_target.png', dpi=300)
plt.show()

# Correlación de la previsiones con la hora ideal
"""
x = datos["hora_ideal"]
y = datos["prev"]
plt.scatter(x, y)
plt.show()
"""
# Preferencias de horarios según grupos de edad

barWidth = 0.25
lista_1 = []
lista_2 = []
lista_3 = []

bars1 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['edad'] > 0) & (new_dataF['edad'] <= 30) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_1.append(bars1)
lista_1.append(bars2)
lista_1.append(bars3)
lista_1.append(bars4)
lista_1.append(bars5)
lista_1.append(bars6)

bars1 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['edad'] >= 31) & (new_dataF['edad'] <= 40) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_2.append(bars1)
lista_2.append(bars2)
lista_2.append(bars3)
lista_2.append(bars4)
lista_2.append(bars5)
lista_2.append(bars6)

bars1 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['edad'] > 40) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_3.append(bars1)
lista_3.append(bars2)
lista_3.append(bars3)
lista_3.append(bars4)
lista_3.append(bars5)
lista_3.append(bars6)

plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
r1 = np.arange(len(lista_1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, lista_1, color='#D97904', width=barWidth, edgecolor='white', label='Rango Etario 1')
plt.bar(r2, lista_2, color='#BF0404', width=barWidth, edgecolor='white', label='Rango Etario 2')
plt.bar(r3, lista_3, color='#730202', width=barWidth, edgecolor='white', label='Rango Etario 3')

plt.xlabel('Categoria de horario', fontweight='bold')
plt.ylabel('Nivel de preferencia')
plt.xticks([r + barWidth for r in range(len(lista_1))], ['Horario A', 'Horario B', 'Horario C', 'Horario D',
                                                         'Horario E', 'Horario F'], rotation=45)

plt.title("Preferencias de horarios según grupos de edad")
plt.legend()
plt.savefig("PreferenciasPorEdad.png", dpi=300)
plt.show()

# Preferencias de horarios para personas con y sin hijos

barWidth = 0.25
lista_1 = []
lista_2 = []

bars1 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['hijos'] == 1) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_1.append(bars1)
lista_1.append(bars2)
lista_1.append(bars3)
lista_1.append(bars4)
lista_1.append(bars5)
lista_1.append(bars6)

bars1 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['hijos'] == 2) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_2.append(bars1)
lista_2.append(bars2)
lista_2.append(bars3)
lista_2.append(bars4)
lista_2.append(bars5)
lista_2.append(bars6)


plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
r1 = np.arange(len(lista_1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, lista_1, color='#D97904', width=barWidth, edgecolor='white', label='Personas con hijo')
plt.bar(r2, lista_2, color='#BF0404', width=barWidth, edgecolor='white', label='Personas sin hijo')

plt.xlabel('Categoria de horario', fontweight='bold')
plt.ylabel('Nivel de preferencia')
plt.xticks([r + barWidth for r in range(len(lista_1))], ['Horario A', 'Horario B', 'Horario C', 'Horario D',
                                                         'Horario E', 'Horario F'], rotation=45)

plt.title("Preferencias de horarios para personas con y sin hijos")
plt.legend()
plt.savefig("PreferenciasConSinHijos.png", dpi=300)
plt.show()

# Preferencias de horarios para personas con y  sin enfermedad cronica

barWidth = 0.25
lista_1 = []
lista_2 = []

bars1 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['enf_cron'] == 1) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_1.append(bars1)
lista_1.append(bars2)
lista_1.append(bars3)
lista_1.append(bars4)
lista_1.append(bars5)
lista_1.append(bars6)

bars1 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['enf_cron'] == 2) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_2.append(bars1)
lista_2.append(bars2)
lista_2.append(bars3)
lista_2.append(bars4)
lista_2.append(bars5)
lista_2.append(bars6)


plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
r1 = np.arange(len(lista_1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, lista_1, color='#D97904', width=barWidth, edgecolor='white', label='Personas con enfermedad cronica')
plt.bar(r2, lista_2, color='#BF0404', width=barWidth, edgecolor='white', label='Personas sin enfermedad cronica')

plt.xlabel('Categoria de horario', fontweight='bold')
plt.ylabel('Nivel de preferencia')
plt.xticks([r + barWidth for r in range(len(lista_1))], ['Horario A', 'Horario B', 'Horario C', 'Horario D',
                                                         'Horario E', 'Horario F'], rotation=45)

plt.title("Preferencias de horarios para personas con y  sin enfermedad cronica")
plt.legend()
plt.savefig("PreferenciasConSinCron.png", dpi=300)
plt.show()

# Preferencias de horarios según frecuencia de asistencia

barWidth = 0.25
lista_1 = []
lista_2 = []
lista_3 = []
lista_4 = []

bars1 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['frec_asistencia'] == 1) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_1.append(bars1)
lista_1.append(bars2)
lista_1.append(bars3)
lista_1.append(bars4)
lista_1.append(bars5)
lista_1.append(bars6)

bars1 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['frec_asistencia'] == 2) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_2.append(bars1)
lista_2.append(bars2)
lista_2.append(bars3)
lista_2.append(bars4)
lista_2.append(bars5)
lista_2.append(bars6)

bars1 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['frec_asistencia'] == 3) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_3.append(bars1)
lista_3.append(bars2)
lista_3.append(bars3)
lista_3.append(bars4)
lista_3.append(bars5)
lista_3.append(bars6)

bars1 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 1)].count()[0]
bars2 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 2)].count()[0]
bars3 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 3)].count()[0]
bars4 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 4)].count()[0]
bars5 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 5)].count()[0]
bars6 = new_dataF[(new_dataF['frec_asistencia'] == 4) & (new_dataF['hora_ideal'] == 0)].count()[0]

lista_4.append(bars1)
lista_4.append(bars2)
lista_4.append(bars3)
lista_4.append(bars4)
lista_4.append(bars5)
lista_4.append(bars6)

plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
r1 = np.arange(len(lista_1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, lista_1, color='#F29F05', width=barWidth, edgecolor='white', label='Menos de 1 vez al año ')
plt.bar(r2, lista_2, color='#D97904', width=barWidth, edgecolor='white', label='1 vez al año ')
plt.bar(r3, lista_3, color='#BF0404', width=barWidth, edgecolor='white', label='2 veces al año')
plt.bar(r4, lista_4, color='#730202', width=barWidth, edgecolor='white', label='Cada mes')

plt.xlabel('Categoria de horario', fontweight='bold')
plt.ylabel('Nivel de preferencia')
plt.xticks([r + barWidth for r in range(len(lista_1))], ['Horario A', 'Horario B', 'Horario C', 'Horario D',
                                                         'Horario E', 'Horario F'], rotation=45)

plt.title("Preferencias de horarios según frecuencia de asistencia")
plt.legend()
plt.savefig("PreferenciasFrecueniaAsis.png", dpi=300)
plt.show()

# matriz de correlación de variables

plt.style.use("ggplot")
d = pd.read_csv("prueba_5.csv")
datos = pd.DataFrame(d)
x = datos.drop(['hora_ideal'], axis=1)
y = datos['hora_ideal']
pd.plotting.scatter_matrix(datos, c=y, figsize=[10, 10], s=150, marker='o')
plt.savefig("ScatterMatrix.png", dpi=300)
plt.show()