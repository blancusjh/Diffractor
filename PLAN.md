# Plan de reorganización: Diffractor v2 + Ground Truth

Dos propósitos, dos paquetes, un contrato entre ellos. La regla organizadora:
**nada entra al motor físico de Diffractor sin haber sido contrastado contra el
ground truth**, y el ground truth a su vez se valida contra soluciones
analíticas exactas. Eso convierte la validación en una cadena con eslabones
explícitos, no en un gesto:

    analítico exacto  →  solver de referencia (BEM/FEM)  →  Diffractor v2
    (bola escalar,        (Müller BOR-BEM,                  (ASM + operador
     interfaz plana)       luego FEM si hace falta)          de interfaz local)

Lo importante: gran parte del eslabón central **ya existe** como prototipo de
esta sesión. No se parte de cero; se parte de código verificado que hay que
endurecer.

---

## Estructura del repositorio

Monorepo con dos paquetes instalables y una suite de benchmarks compartida:

```
diffractor/                      # el producto (propósito I)
  src/diffractor/
    core/          # Field, Grid, espectro angular, Hankel/FFT — sin física de medios
    geometry/      # SOLO geometría: curvas, superficies de revolución, normales,
                   #   curvaturas, parametrizaciones exactas (óvalo de Descartes,
                   #   esfera, cónicas). Nada de medios ni interfaces.
    optics/        # media.py: Medium (una clase, un índice — no un subsistema)
                   # interfaces.py: Interface = lugar geométrico Σ + dos Medium.
                   #   Medium 1 — Σ — Medium 2. Nada más: sin Fresnel, sin OPL.
    scattering/    # la respuesta de la interfaz: fresnel.py (t_s escalar),
                   #   planar.py (operador espectral exacto), y el operador
                   #   local sobre Σ (fase 3)
    propagation/   # transporte por el espacio: exact.py (ASM/RS1/Hankel),
                   #   paraxial.py (Fresnel con GATE), transport.py (factores
                   #   GENERALES de conservación de energía: tubos de rayos,
                   #   flujo n·cosθ; el caso estigmático P = t_s·d2/d1 vive aquí)
    sources/       # onda plana, fuente puntual (esférica con 1/r), gaussiana
    analysis/      # mediciones: OPL/OPD (opl.py), pupila demodulada en esferas
                   #   de referencia, energía, PSF
    viz/
  tests/           # unitarios + regresión contra benchmarks/

groundtruth/                     # el solver de referencia (propósito II)
  src/groundtruth/
    exact/         # soluciones cerradas: bola escalar (Mie escalar), interfaz plana,
                   #   identidades (R+T=1, Snell, w2/w1)
    bem/           # Müller BOR-BEM (m=0 hoy; m≠0 si algún día hay fuera de eje)
    fem/           # (fase posterior, solo si BEM encuentra un límite real)
    protocol/      # el protocolo de medición: demodulación en S2, presupuesto
                   #   de energía, barrido en kR y extrapolación 1/(kR)
  tests/

benchmarks/                      # los casos canónicos, con resultados congelados
  cases/           # definición de cada caso (geometría, λ, conjugados, NA)
  golden/          # resultados de referencia versionados (npz + tolerancias)
  runner.py        # corre Diffractor contra golden y reporta desviaciones
```

Por qué monorepo: el contrato entre los dos paquetes (formato de escena, formato
de resultado de pupila) cambia junto; separarlos en repos ahora solo añade
fricción. `groundtruth` **no** importa `diffractor` — solo comparten los tipos
de `benchmarks/`.

---

## Propósito II primero (el ground truth manda)

Aunque el objetivo final es Diffractor, el orden correcto es endurecer primero
la referencia, porque cada decisión del operador de interfaz del propósito I se
va a juzgar contra ella. Estado real y lo que falta:

**Ya verificado en esta sesión** (entra casi tal cual a `groundtruth/`):

- `exact/scalar_ball` — bola dieléctrica con condiciones escalares
  ([ψ]=0, [∂ψ/∂n]=0). Residuos de contorno ~1e-15, flujo ~1e-17, límite de
  interfaz plana R+T−1 = 2e-16. **Esto es exactamente la "teoría de Mie
  escalar" que mencionas** — Mie EM no sirve aquí, como bien dijiste, pero la
  separación de variables con las condiciones escalares sí, y ya está.
- `bem/muller` — BOR-BEM de segunda especie. cond(M) ≈ 170 plana en N,
  convergencia O(h²) limpia, campo interior a 3.7e-5 contra la bola exacta.
  Bug de signo en K documentado y corregido; formas expm1 estables para los
  núcleos diferencia.
- `protocol/` — la medición de pupila en S2 (demodulación de portadora
  convergente), ya ejercitada en el ovoide: fase plana a 0.03 λ, ratio absoluto
  1.0025, factor borde/eje 0.213 vs 0.219 a k₂R₂=31.

**Deuda técnica concreta, en orden:**

1. **Velocidad del ensamblado.** El lazo de banda cercana es Python puro y es
   lo que hizo abortar las corridas de zi=16λ. Vectorizarlo (o Numba). Meta:
   ovoide a 200–500λ en minutos. Sin esto no hay barrido en kR decente.
2. **Cerrar el barrido en kR.** Hoy hay dos puntos (k₂R₂ = 21, 31); el factor
   de apodización está confirmado solo al ~13%. Con 4 puntos se demuestra (o
   refuta) la caída 1/(kR) del residuo, que es lo que licencia extrapolar al
   sistema real de 17 000λ.
3. **Explicar el rizado.** El scatter del 11% a kR bajo es convergido en malla
   (verificado), o sea físico — probablemente difracción de borde del casquete.
   Hay que separarlo: correr el mismo cuerpo con apertura apodizada suave y ver
   si desaparece.
4. **FEM: solo si hace falta.** Un FEM volumétrico 3D no llega ni cerca de
   estas escalas; un FEM axisimétrico con PML duplicaría lo que el BEM ya hace.
   Propongo dejarlo como *contingencia* — se activa únicamente si aparece un
   caso que el BEM no pueda representar (medios inhomogéneos, por ejemplo).
   No construir dos ground truths en paralelo.

---

## Propósito I: Diffractor v2

### Qué se elimina y qué se conserva

Fuera del núcleo físico: `thin_lens`, `Surface.phase_mask` como modelo de
superficie (el error de proyección es primer orden en el sag — 543 λ RMS a
NA 0.75), y el promedio no físico (Ts+Tp)/2. Si quieres conservar la lente
delgada por conveniencia pedagógica, que viva en un módulo `ideal/`
explícitamente rotulado como no físico y fuera de `scattering/` — decisión
tuya; mi voto es eliminarla y que el caso "lente ideal" se construya como
pupila sintética en `analysis/`.

Se conserva y se promueve: los propagadores exactos (ASM/RS1/Hankel), la grilla
de salida desacoplada (eso está bien hecho en el Diffractor actual), y la
parametrización cerrada del ovoide (reemplaza al Newton del sag, que muere en
el borde y capaba la NA a 0.37 cuando la superficie soporta 0.94).

### El operador de interfaz (tu propuesta, formalizada)

Tu idea — aproximación local sobre la interfaz, k local proyectado para
obtener ángulos, transferencia plano→interfaz, factor de Fresnel escalar — es
esencialmente una **aproximación de plano tangente local** (local plane-wave /
tangent-plane approximation, pariente de la aproximación de Kirchhoff para
superficies). Formalizada en tres pasos, cada uno con su error controlado:

**(a) Transferencia plano → interfaz.** El dato viene en un plano z = z₀; la
interfaz es Σ: z = ζ(ρ). Para llevar el campo al punto Q = (ρ, ζ(ρ)) se usa el
propagador exacto del medio 1 evaluado en superficie no plana:

    ψ(ρ, ζ(ρ)) = ∫ ψ̃(k⊥) exp(i k⊥·ρ + i k_z(k⊥) ζ(ρ)) d²k⊥

Esto NO es la aproximación de proyección (que pone ζ dentro de una fase fija
(n₁−n₂)k₀ζ): aquí cada componente espectral lleva su k_z verdadero. Es exacto
mientras Σ no pliegue el frente (una componente evanescente o una interfaz
multivaluada en z lo rompen). Costo: una NUFFT o evaluación por anillos en el
caso axisimétrico.

**(b) Refracción local.** En Q, el campo local se trata como onda plana con
vector k_loc = ∇φ (fase local, WKB). Se proyecta sobre la normal n̂(Q): eso da
i₁; Snell da i₂; el coeficiente es **t_s del problema escalar**,

    t_s = 2 n₁ cos i₁ / (n₁ cos i₁ + n₂ cos i₂),

y el k transmitido se reconstruye conservando la componente tangencial
(k_t continua, |k₂| = n₂k₀). Nota de esta sesión que hay que respetar: t_s a
nivel de amplitud del CAMPO; si en algún punto se trabaja con potencias, la
conversión es T_s = (n₂cos i₂/n₁cos i₁)|t_s|², y jamás el promedio con t_p.

**(c) Interfaz → plano de salida.** El campo transmitido, conocido ahora sobre
Σ con su k local, se lleva a un plano en el medio 2. Dos opciones a evaluar:
re-expansión en espectro angular del medio 2 (inversa de (a)), o integral de
Kirchhoff desde Σ con el kernel exacto de k₂ (más cara pero sin hipótesis
adicional; ya la tenemos escrita del ovoide).

**Validez esperada:** exacta para interfaz plana (los tres pasos colapsan al
operador diagonal ψ̃₂ = t(k⊥)ψ̃₁ — que debe implementarse aparte como
primitiva, porque ahí no hay aproximación ninguna); error que crece con
curvatura/λ y cerca de incidencia rasante (donde t_s varía rápido dentro de la
"huella" local de la onda). El escenario del ovoide a NA 0.75 con i₁ hasta 80°
es deliberadamente el caso difícil — perfecto como benchmark de estrés.

### La escalera de validación del operador

Cada peldaño aísla un ingrediente; no se sube al siguiente sin pasar el actual:

| # | Caso | Referencia | Qué aísla |
|---|------|-----------|-----------|
| 1 | Interfaz plana, incidencia oblicua | operador espectral exacto | pasos (b)+(c) sin curvatura; debe ser exacto a máquina |
| 2 | Interfaz plana, fuente puntual | RS1 + t(k⊥) exacto | espectro ancho, sin curvatura |
| 3 | Casquete esférico suave (sag ≪ zi) | BEM + bola exacta | curvatura pequeña; primer test de (a) |
| 4 | Ovoide estigmático NA 0.3–0.75 | fórmula P = t_s·d₂/d₁ + BEM | curvatura fuerte, i₁ hasta 80°; fase debe salir plana |
| 5 | Esfera completa (caústica) | bola escalar exacta | régimen donde los rayos fallan; mide degradación honesta |

El peldaño 4 tiene doble referencia: la analítica (válida en el límite kR→∞) y
el BEM (exacto a kR finito) — la diferencia entre ambas ya la medimos y es la
física de difracción, no error.

### Métricas y contrato de comparación

Toda comparación usa el protocolo de `groundtruth/protocol/`: pupila medida en
esfera de referencia (amplitud, fase demodulada), energía total a través de S2,
y PSF en el plano focal. Tolerancias por peldaño versionadas en
`benchmarks/golden/`. La CI corre los peldaños 1–3 siempre (baratos); 4–5 en
manual/nightly.

---

## Fases y orden de trabajo

**Fase 0 — consolidar (corto).** Mover el código de esta sesión al monorepo:
`ovoid_geometry`, `scalar_ball`, `bem_bor`, `bem_muller`, `ovoid_body`,
protocolo de medición, y los scripts de verificación convertidos en tests.
Congelar los resultados actuales como primeros `golden/`.

**Fase 1 — endurecer ground truth.** [HECHA — con un resultado negativo
importante.]

*Logrado:* solver ~5x más rápido y mejor calibrado. Los defaults se fijaron
midiendo contra la bola exacta, no adivinando: n_phi = 24 (los núcleos
diferencia son suaves en phi; 64 era desperdicio puro), ng_smooth = 1
(¡punto medio es 3.6x MÁS preciso que Gauss-2, porque punto-medio con
colocación en punto medio tiene cancelación de error par que Gauss rompe!),
n_near = 5 (quita el suelo de convergencia y cuesta O(N), no O(N^2)).

*Resultado negativo:* el barrido en kR reveló que **el cuerpo cerrado del
benchmark es un resonador**. Al cerrar el ovoide con la mortaja se crea una
cavidad dieléctrica sin pérdidas. Prueba decisiva: variando solo la longitud
de la mortaja — parámetro río abajo, fuera del cono convergente, que no puede
afectar la física — el factor borde/eje medido oscila +/-38 %.

Consecuencia: **se retracta** la afirmación de que la medida BEM confirmaba
P = t_s d2/d1 al 0.25 %/2.8 %. Aquella medida estaba contaminada. El test de
convergencia de malla que le dio confianza verificaba la discretización, no
la geometría: convergía impecablemente a la respuesta equivocada.

*Diagnóstico:* campo reverberante (como el de una sala). Suma energía
(+20 %, estable bajo promediado) y domina donde el campo directo es más
débil — el borde de la pupila. Un modelo de fondo constante ajustado solo
con el exceso de energía recupera la mitad del hueco (2.18x -> 1.47x), luego
el campo atrapado no es isótropo. El promediado sobre modos no lo rescata:
converge a 0.48, no a 0.219.

*Observable robusto:* la energía a través de S2 sí converge bajo promediado
(1.20 +/- 0.01 x la predicción). Las cantidades integradas sobreviven; el
factor puntual en el borde no.

**Fase 1b — benchmark abierto (NUEVA, bloquea la escalera).** El problema
físico tiene Omega2 no acotado; cerrarlo es lo que introduce la cavidad. Hay
que formular el problema con superficie ABIERTA (solo el casquete, con
condición de radiación en ambos semiespacios). Las incógnitas pasan a ser los
SALTOS a través de la superficie y aparece la singularidad de borde ~rho^(1/2)
en el reborde. Es más trabajo, pero además es el modelo físicamente correcto
de una lente con diafragma: el reborde difracta de verdad.

Alternativa a evaluar antes de construirlo: cerrar con una terminación que no
resuene. La absorción no sirve — la aritmética dice que para amortiguar la
cavidad haría falta ~20 % de pérdida en el camino directo.

**Fase 2 — esqueleto Diffractor v2.** `core/`, `media/`, `geometry/`,
`propagation/` con lo que ya es correcto. El gate de Fresnel: cada llamada
comprueba z ≫ [πr⁴/4λ]^(1/3) y rehúsa (o degrada a warning explícito) si no se
cumple. El thin-element muere aquí.

**Fase 3 — operador de interfaz.** Primero la primitiva exacta plana
(peldaños 1–2), después la versión local sobre Σ (peldaños 3–4). Es la fase
larga y donde tu idea se juega; las decisiones de (c) se toman con datos de la
escalera, no a priori.

**Fase 4 — integración.** Escenas multi-interfaz (encadenar el operador),
ejemplos migrados, documentación con la cadena de validación visible.

Dependencias: F1 y F2 pueden ir en paralelo; F3 requiere F1 (tolerancias) y
F2 (esqueleto); F4 requiere F3.

---

## Riesgos conocidos (de esta sesión, no especulativos)

- El paso (a) con ζ grande mete componentes casi rasantes donde k_z→0: ahí la
  transferencia es exacta pero mal condicionada. Puede necesitar el split
  casquete-por-anillos con planos tangentes locales en vez de un plano global.
- El rizado físico a kR moderado significa que Diffractor v2 *no debe*
  reproducir exactamente al BEM a escalas pequeñas — el criterio de éxito es
  converger a él como 1/(kR), no igualarlo punto a punto.
- t_s varía O(1) en la última fracción del pupila (i₁ 70°→90°): la "huella"
  local allí es grande comparada con la escala de variación de t_s, y es donde
  la aproximación local se degradará primero. El peldaño 4 lo medirá.
