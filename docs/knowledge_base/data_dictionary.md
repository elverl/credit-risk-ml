---
kb_version: 1
status: current
scope: data_dictionary
model_version: v2
---

# Diccionario de datos

## Reglas aprobadas

- En features documentadas como binarias: `1` significa que la característica
  está presente y `0` que no está presente. No aplicar esta regla a variables
  continuas.
- `-1` codifica cliente sin tarjeta de crédito cuando sea aplicable a una
  variable relacionada con tarjeta. No es un valor económico negativo ni debe
  extenderse automáticamente a otras variables.
- Cuando una propiedad no aparece en este documento, se considera **No
  documentada en el proyecto**.

## Features en orden contractual

| # | Nombre técnico | Descripción | Tipo | Ventana | Unidad/escala | Codificación y valores especiales | Observaciones/limitaciones |
|---:|---|---|---|---|---|---|---|
| 1 | `n_atr_1d_6` | Número de atrasos mayores a 1 día | Conteo | Últimos 6 meses | Conteo; dominio no documentado | Valores especiales no documentados | Fuente operacional y tratamiento de historia ausente no documentados |
| 2 | `cl_9_1.0` | Indicador de máxima calificación CPP | Binaria | Últimos 9 meses | 0/1 | 1 = presente; 0 = no presente | Fuente y tratamiento de calificación desconocida no documentados |
| 3 | `cl_12_1.0` | Indicador de máxima calificación CPP | Binaria | Últimos 12 meses | 0/1 | 1 = presente; 0 = no presente | Fuente y tratamiento de calificación desconocida no documentados |
| 4 | `cl_18_2.0` | Indicador de máxima calificación Deficiente | Binaria | Últimos 18 meses | 0/1 | 1 = presente; 0 = no presente | Fuente y tratamiento de calificación desconocida no documentados |
| 5 | `cl_9_3.0` | Indicador de máxima calificación Dudoso | Binaria | Últimos 9 meses | 0/1 | 1 = presente; 0 = no presente | Fuente y tratamiento de calificación desconocida no documentados |
| 6 | `ind_vjrc_36_1` | Indicador de crédito vencido, refinanciado o castigado | Binaria | Últimos 36 meses | 0/1 | 1 = presente; 0 = no presente | Definición operacional de cada evento y fuente no documentadas |
| 7 | `ind_per_x9_1.0` | Indicador de deuda bancaria personal consecutiva | Binaria | Últimos 9 meses | 0/1 | 1 = presente; 0 = no presente | Significado operacional de “consecutiva” y fuente no documentados |
| 8 | `prc_u_tc_cns` | Máximo porcentaje de utilización de tarjetas de crédito de consumo | Continua | Ventana exacta no documentada | Proporción decimal; 50% = `0.5` | Aplicabilidad de `-1` no confirmada específicamente | Fórmula y denominador no documentados |
| 9 | `tas_u_tc` | Porcentaje de utilización de tarjetas de crédito en el último mes | Continua | Último mes | Proporción decimal; 50% = `0.5` | `-1` = cliente sin tarjeta de crédito | `-1` no significa -100%; fórmula y denominador no documentados |
| 10 | `c_tc_mn_24` | Cantidad mínima de tarjetas de crédito | Conteo | Últimos 24 meses | Conteo; dominio no documentado | Relacionada con tarjetas; aplicabilidad exacta de `-1` no documentada | Agregación temporal y meses sin observación no documentados |
| 11 | `c_tc_mn_24_sld` | Cantidad mínima de tarjetas con saldo mayor a 0 | Conteo | Últimos 24 meses | Conteo; dominio no documentado | Relacionada con tarjetas; aplicabilidad exacta de `-1` no documentada | Definición operacional de saldo y agregación no documentadas |
| 12 | `c_pcns` | Número de créditos de consumo al momento de evaluación | Conteo | Momento de evaluación | Conteo; dominio no documentado | Valores especiales no documentados | Estados de crédito incluidos y fuente no documentados |
| 13 | `vr_s_t_36` | Variación porcentual del saldo de deuda respecto a 36 meses atrás | Continua | Respecto a 36 meses atrás | Proporción decimal; caída de 73.66% = `-0.7366` | Valores especiales no documentados | Fórmula, denominador cero y clipping no documentados |
| 14 | `mx_ltc_36` | Máximo monto de línea de tarjeta de crédito | Continua monetaria | Últimos 36 meses | PEN, soles peruanos | Relacionada con tarjetas; aplicabilidad exacta de `-1` no documentada | Agregación entre entidades y casos sin línea no documentados |
| 15 | `ipc_t6` | Tasa de inflación anual de hace 6 meses | Continua | Tasa anual observada 6 meses antes | Proporción decimal; 3.25% = `0.0325` | Valores especiales no documentados | Fuente del índice, país, base y vintage no documentados |

## Target

| Nombre | Tipo | Clase positiva | Clase negativa | Ventana de performance | Limitaciones semánticas |
|---|---|---|---|---|---|
| `incumplimiento` | Binaria | `1`: alcanza 30+ DPD dentro de los 12 meses posteriores a originación | `0`: no alcanza 30+ DPD en esa ventana | 12 meses desde la originación | Reglas de cure, write-off y otras definiciones no están documentadas |

## Nota sobre `-1`

La regla aprobada es condicional: aplica cuando `-1` se usa en una variable
relacionada con tarjeta de crédito. Solo `tas_u_tc` tiene aplicación específica
confirmada en esta versión. Para las demás variables relacionadas con tarjeta,
la aplicabilidad exacta debe confirmarse antes de interpretar el código.

