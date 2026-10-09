# MEIKA Vox en Colab

1. Abre el [cuaderno público](https://colab.research.google.com/github/MeikaLab/meika-vox/blob/main/notebooks/MEIKA_Vox_Colab.ipynb).
2. Selecciona GPU en **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, si está disponible.
3. Pulsa **Ejecutar todas**, espera la preparación y autoriza Drive.
4. En el panel navega hasta una carpeta concreta o escribe su ruta desde Mi unidad. Pulsa **Buscar audios aquí**. Todos quedan seleccionados; Ctrl/Cmd permite cambiar la selección.
5. Escribe el proyecto, confirma idioma y pulsa **Transcribir seleccionados**.
6. Al terminar, descarga el ZIP o busca el proyecto en Drive. Los TXT están en `MEIKA_Vox/Proyectos/<proyecto>/Transcripciones/Lectura`.

El enlace del panel busca el nombre del proyecto en Drive; no es un enlace directo a una carpeta determinada. El navegador de esta versión trabaja dentro de **Mi unidad**. Las unidades compartidas y los enlaces arbitrarios de carpetas no están implementados.

## Calidad y voces

Con GPU se utiliza WhisperX **large-v3**. Sin GPU, el panel exige aceptar expresamente el modo **CPU + small**; cambia la velocidad y puede cambiar la calidad. No se cambia de modelo silenciosamente. Si falta memoria durante el reconocimiento, se reduce el batch interno manteniendo el modelo. Si sigue fallando, el audio queda pendiente.

**Distinguir hablantes** agrega etiquetas de voz; no identifica personas y puede fallar con solapamientos. Para activarlo:

1. Crea una cuenta gratuita en [Hugging Face](https://huggingface.co/join).
2. Acepta las condiciones de [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Genera una clave con permisos de lectura en [Tokens](https://huggingface.co/settings/tokens).
4. En Colab abre **Secretos** (llave), crea `HF_TOKEN`, pega la clave y permite acceso al cuaderno.
5. Marca la opción y pulsa **Comprobar acceso**. La primera descarga puede tardar.

La clave se lee desde Secretos y no se escribe en el cuaderno ni en los resultados. Un fallo inicial de acceso bloquea el lote hasta corregirlo o desactivar la opción. Si la separación falla después de transcribir, se conserva el texto con una observación y voces sin identificar. Para generar otra versión con voces, selecciona esos audios y activa **Crear una nueva versión**; esta versión vuelve a ejecutar el reconocimiento.

pyannote es el único motor de voces conectado. Nemotron y Sherpa-ONNX se describen como candidatos; no son opciones ejecutables ni existe una comparación de calidad validada en el proyecto.

## Progreso y recuperación

La barra mide archivos revisados, incluidos los que fallan. El balance distingue guardados, ya existentes, errores y archivos aún sin procesar. La etapa y el reloj muestran actividad; no son un porcentaje interno ni una estimación de tiempo restante.

**Detener después de este audio** guarda el resultado actual antes de parar. **Reintentar pendientes** procesa errores y archivos no iniciados. Si Colab pierde su sesión, vuelve a ejecutar todo y selecciona el mismo proyecto, audios y configuración. Se omiten resultados íntegros y se recuperan manifiestos válidos si se perdió el índice. El audio que estaba en curso puede repetirse: no hay reanudación interna por fragmentos.

Los cambios visuales, la ruta del ejecutable y el tamaño del batch no invalidan los resultados. Cambiar el modelo, idioma, glosario o solicitar voces sí puede producir otra ejecución. La migración de índices antiguos adopta solo resultados verificables del mismo proyecto/modelo/idioma, con el mismo estado de voces y sin normalizaciones; en casos ambiguos se conserva la versión anterior y se genera otra.

Los modelos se reutilizan durante un lote y se liberan al terminar. Los audios se copian temporalmente a la máquina de Colab; los originales de Drive se conservan. Los paquetes técnicos de cada ejecución mantienen identificadores distintos, comprobaciones de integridad, texto original, tiempos y controles QA. Los TXT de Lectura representan la última versión generada de cada grabación.

## Alcance de la validación

Los tests automáticos verifican contratos, integridad, selección, guardado, reanudación y comportamiento ante fallos con datos sintéticos. No acreditan precisión de WhisperX, separación de voces ni disponibilidad real de una GPU de Colab. Antes de utilizar citas, revisa el audio y el texto. El glosario normaliza después del reconocimiento; no es vocabulario contextual enviado al modelo.

Colab gratuito no garantiza GPU ni duración. Consulta las [condiciones y límites oficiales](https://research.google.com/colaboratory/faq.html). La ejecución ocurre en Google, y autorizar Drive permite al código del cuaderno acceder a sus archivos. No se publican grabaciones en GitHub por utilizar esta herramienta.
