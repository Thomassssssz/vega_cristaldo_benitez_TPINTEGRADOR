# Trabajo Práctico Integrador de Estadística
# Bienestar digital: uso responsable de pantallas
# Librerías requeridas: pandas, numpy y matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLOR_VERDE = "#2E7D32"
COLOR_VERDE_CLARO = "#A5D6A7"
COLOR_VERDE_OSCURO = "#1B5E20"

# ----------------------------------------------------------------------
# 1. Carga de archivos
# ----------------------------------------------------------------------
archivo_muestra = "TPINTEGRADOR-vega-cristaldo-benitez.xlsx"
archivo_catedra = "TrabajoIntegrador-Estadistica.xlsx"

muestra = pd.read_excel(archivo_muestra, sheet_name="Datos_limpios_200")
catedra = pd.read_excel(archivo_catedra, sheet_name="Datos Brutos", header=2)

# Se renombran columnas para facilitar el trabajo.
muestra = muestra.rename(columns={
    "Horas pantalla promedio": "horas_pantalla",
    "Horas redes sociales": "horas_redes",
    "Rendimiento académico": "rendimiento"
})

catedra = catedra.rename(columns={
    "Horas de  consumo de redes sociales por día": "horas_redes",
    "Rendimieno académico": "rendimiento"
})

# Se eliminan filas vacías en caso de existir.
muestra = muestra.dropna(subset=["horas_pantalla", "horas_redes", "rendimiento"])
catedra = catedra.dropna(subset=["horas_redes", "rendimiento"])

# ----------------------------------------------------------------------
# 2. Funciones auxiliares
# ----------------------------------------------------------------------
def tabla_frecuencias(datos, columna):
    """Construye tabla de frecuencias para una variable discreta."""
    frecuencia = datos[columna].value_counts().sort_index()
    tabla = pd.DataFrame({
        "xi": frecuencia.index,
        "fi": frecuencia.values
    })
    tabla["Fi"] = tabla["fi"].cumsum()
    tabla["fri"] = tabla["fi"] / tabla["fi"].sum()
    tabla["Fri"] = tabla["fri"].cumsum()
    tabla["pi %"] = tabla["fri"] * 100
    tabla["Pi %"] = tabla["Fri"] * 100
    return tabla


def medidas_estadisticas(datos, columna):
    """Calcula medidas estadísticas principales."""
    serie = datos[columna]
    return {
        "media": serie.mean(),
        "mediana": serie.median(),
        "moda": serie.mode().iloc[0],
        "varianza": serie.var(ddof=1),
        "desvio_estandar": serie.std(ddof=1),
        "coeficiente_variacion": serie.std(ddof=1) / serie.mean() * 100,
        "asimetria": serie.skew(),
        "curtosis": serie.kurt()
    }


def regresion_lineal(datos, x_col, y_col):
    """Calcula recta de regresión, correlación y determinación."""
    x = datos[x_col].to_numpy()
    y = datos[y_col].to_numpy()
    pendiente, ordenada = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2
    return pendiente, ordenada, r, r2

# ----------------------------------------------------------------------
# 3. Análisis de la muestra propia
# ----------------------------------------------------------------------
print("ANÁLISIS DE LA MUESTRA PROPIA")
print("Cantidad de casos:", len(muestra))
print("\nMedidas de horas de pantalla:")
print(medidas_estadisticas(muestra, "horas_pantalla"))

# Histograma de horas de pantalla.
plt.figure(figsize=(8, 5))
plt.hist(muestra["horas_pantalla"], bins=10, color=COLOR_VERDE, edgecolor=COLOR_VERDE_OSCURO)
plt.title("Histograma - Horas promedio de pantalla por día")
plt.xlabel("Horas promedio de pantalla por día")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("histograma_pantalla.png", dpi=200)
plt.show()

# ----------------------------------------------------------------------
# 4. Contrastación con la base de la cátedra
# ----------------------------------------------------------------------
print("\nANÁLISIS DE LA BASE PROPORCIONADA POR LA CÁTEDRA")
print("Cantidad de casos:", len(catedra))

frecuencias_catedra = tabla_frecuencias(catedra, "horas_redes")
print("\nTabla de frecuencias - Base de la cátedra")
print(frecuencias_catedra)

medidas_catedra = medidas_estadisticas(catedra, "horas_redes")
print("\nMedidas de horas de redes sociales - Base de la cátedra")
print(medidas_catedra)

media_muestra_redes = muestra["horas_redes"].mean()
media_catedra_redes = catedra["horas_redes"].mean()
diferencia_medias = media_catedra_redes - media_muestra_redes

print("\nCOMPARACIÓN DE MEDIAS")
print("Media muestra propia:", round(media_muestra_redes, 2))
print("Media base cátedra:", round(media_catedra_redes, 2))
print("Diferencia:", round(diferencia_medias, 2))

# Gráfico de frecuencias de la base de la cátedra.
plt.figure(figsize=(8, 5))
plt.bar(frecuencias_catedra["xi"], frecuencias_catedra["fi"], color=COLOR_VERDE, edgecolor=COLOR_VERDE_OSCURO)
plt.title("Horas de redes sociales por día - Base de la cátedra")
plt.xlabel("Horas por día")
plt.ylabel("Frecuencia absoluta")
plt.xticks(frecuencias_catedra["xi"])
plt.tight_layout()
plt.savefig("frecuencia_catedra.png", dpi=200)
plt.show()

# Gráfico comparativo de medias.
plt.figure(figsize=(6, 4))
plt.bar(["Muestra propia", "Base cátedra"], [media_muestra_redes, media_catedra_redes],
        color=[COLOR_VERDE_CLARO, COLOR_VERDE], edgecolor=COLOR_VERDE_OSCURO)
plt.title("Comparación de media de horas en redes sociales")
plt.ylabel("Horas promedio por día")
plt.tight_layout()
plt.savefig("comparacion_medias.png", dpi=200)
plt.show()

# ----------------------------------------------------------------------
# 5. Regresión lineal
# ----------------------------------------------------------------------
pendiente_m, ordenada_m, r_m, r2_m = regresion_lineal(muestra, "horas_redes", "rendimiento")
pendiente_c, ordenada_c, r_c, r2_c = regresion_lineal(catedra, "horas_redes", "rendimiento")

print("\nREGRESIÓN LINEAL - MUESTRA PROPIA")
print(f"Recta: y = {pendiente_m:.3f}x + {ordenada_m:.3f}")
print(f"Coeficiente de correlación r = {r_m:.3f}")
print(f"Coeficiente de determinación R² = {r2_m:.3f}")

print("\nREGRESIÓN LINEAL - BASE DE LA CÁTEDRA")
print(f"Recta: y = {pendiente_c:.3f}x + {ordenada_c:.3f}")
print(f"Coeficiente de correlación r = {r_c:.3f}")
print(f"Coeficiente de determinación R² = {r2_c:.3f}")

# Diagrama de dispersión de la base de la cátedra.
x = catedra["horas_redes"]
y = catedra["rendimiento"]
recta = pendiente_c * x + ordenada_c

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color=COLOR_VERDE_CLARO, edgecolor=COLOR_VERDE_OSCURO, alpha=0.7)
plt.plot(x, recta, color=COLOR_VERDE_OSCURO, linewidth=2)
plt.title("Regresión lineal - Base de la cátedra")
plt.xlabel("Horas de redes sociales por día")
plt.ylabel("Rendimiento académico")
plt.tight_layout()
plt.savefig("regresion_catedra.png", dpi=200)
plt.show()
