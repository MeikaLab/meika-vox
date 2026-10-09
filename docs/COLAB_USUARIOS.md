# MEIKA Vox en Colab

1. Abre el [cuaderno público](https://colab.research.google.com/github/MeikaLab/meika-vox/blob/main/notebooks/MEIKA_Vox_Colab.ipynb).
2. Selecciona GPU (T4 si aparece) en **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, si está disponible.
3. Pulsa ▶ en **Preparar y abrir MEIKA Vox**, justo debajo de las instrucciones. Espera y autoriza Drive; el panel aparece en esa misma celda, sin bajar al final.
4. Si no sabes dónde están los audios, pulsa **Encontrar carpetas con audios**: revisa Mi unidad y muestra carpetas con cantidades. Verás el avance y podrás detenerla. La búsqueda tiene un límite de 60 segundos o 3.000 carpetas; si se detiene, alcanza el límite o no puede leer carpetas, muestra **Búsqueda parcial**. Los conteos son archivos directos de cada carpeta. Elige un resultado en **Con audios**. También puedes elegir una carpeta en **Abrir carpeta**. Se abre y busca automáticamente, incluyendo subcarpetas. Para volver, pulsa **Volver a la carpeta anterior**. Si está vacía, el panel explica qué archivos ve. **Actualizar audios** repite la búsqueda si agregaste archivos a Drive.
5. Pulsa **Transcribir N audios**. El proyecto toma el nombre de la carpeta y el idioma es español. Puedes excluir audios mediante casillas y elegir la modalidad de voces directamente en el panel y cambiar idioma o glosario en **Opciones adicionales**. También puedes escribir una ruta exacta en la sección opcional y pulsar **Abrir ruta**.
6. Debajo aparece el progreso y, al terminar, la vista previa del primer texto, la descarga del ZIP y los reintentos si hubo errores. Los TXT quedan en `MEIKA_Vox/Proyectos/<proyecto>/Transcripciones/Lectura`.

Se admiten MP3, M4A, WAV, FLAC, OGG, AAC, OPUS, WMA, AIFF y contenedores MP4, WEBM, 3GP y MKV con audio. Un archivo de video sin pista de audio fallará durante la comprobación y quedará pendiente. Los accesos directos de Drive y enlaces a carpetas compartidas pueden requerir una carpeta accesible dentro de Mi unidad.

El enlace del panel busca el nombre del proyecto en Drive; no es un enlace directo a una carpeta determinada. El navegador de esta versión trabaja dentro de **Mi unidad**. Las unidades compartidas y los enlaces arbitrarios de carpetas no están implementados.

## Calidad y voces

Con GPU se utiliza WhisperX **large-v3**. Sin GPU, el panel exige aceptar expresamente el modo **CPU + small**; cambia la velocidad y puede cambiar la calidad. No se cambia de modelo silenciosamente. Si falta memoria durante el reconocimiento, se reduce el batch interno manteniendo el modelo. Si sigue fallando, el audio queda pendiente.

El panel presenta tres modalidades en este orden:

- **Solo transcribir · sin token:** texto con tiempos, sin atribuir voces. Seleccionada por defecto.
- **Separar hablantes · sin token (experimental):** Sherpa-ONNX descarga modelos públicos la primera vez; no pide cuenta de Hugging Face. Su calidad en español está pendiente de validar.
- **Separar hablantes · con token:** pyannote, con instrucciones de acceso visibles únicamente al elegirlo.

**Separar hablantes** agrega etiquetas de voz; no identifica personas y puede fallar con solapamientos. Para activarlo:

1. Crea una cuenta gratuita en [Hugging Face](https://huggingface.co/join).
2. Acepta las condiciones de [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Genera una clave con permisos de lectura en [Tokens](https://huggingface.co/settings/tokens).
4. En Colab abre **Secretos** (llave), crea `HF_TOKEN`, pega la clave y permite acceso al cuaderno.
5. Elige **Separar hablantes · con token** y pulsa **Comprobar acceso**. La primera descarga puede tardar.

La clave se lee desde Secretos y no se escribe en el cuaderno ni en los resultados. Un fallo inicial de acceso bloquea el lote hasta corregirlo o desactivar la opción. Si la separación falla después de transcribir, se conserva el texto con una observación y voces sin identificar. Para generar otra versión con voces, selecciona esos audios y activa **Crear una nueva versión**; esta versión vuelve a ejecutar el reconocimiento.

Sherpa y pyannote están conectados al mismo reconocimiento WhisperX. Sherpa ejecuta las voces en CPU, aunque WhisperX use GPU. En **N.º de voces**, 0 significa detección automática; puedes indicar el número exacto si lo conoces. No hay una comparación de calidad validada que demuestre cuál es mejor. Nemotron sigue fuera del panel.

El modelo de segmentación ONNX de Sherpa deriva de pyannote 3.0; no es Community-1. Se combina con `nemo_en_titanet_small.onnx`, descargado de las publicaciones oficiales de Sherpa. La licencia del paquete se conserva con el modelo en la caché. Fuentes: [modelos](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html) y [API Python](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/python.html).

## Progreso y recuperación

La barra mide archivos revisados, incluidos los que fallan. El balance distingue guardados, ya existentes, errores y archivos aún sin procesar. La etapa indica la operación actual. El reloj muestra cuánto tiempo lleva el lote, incluso durante una espera: no prueba que el audio avance. **Actualizar estado** vuelve a mostrar el estado interno. Si pasan 60 segundos sin nueva etapa, aparece un aviso de espera; no se declara un fallo solo por ello. Las descargas de Sherpa muestran MB recibidos y la separación de voces muestra su porcentaje cuando el motor lo informa. La barra de lote sigue midiendo archivos, no minutos de audio.

**Detener después de este audio** guarda el resultado actual antes de parar. **Reintentar pendientes** procesa errores y archivos no iniciados. Si Colab pierde su sesión, vuelve a ejecutar todo y selecciona el mismo proyecto, audios y configuración. Se omiten resultados íntegros y se recuperan manifiestos válidos si se perdió el índice. El audio que estaba en curso puede repetirse: no hay reanudación interna por fragmentos.

Los cambios visuales, la ruta del ejecutable y el tamaño del batch no invalidan los resultados. Cambiar el modelo, idioma, glosario, motor de voces o número de voces sí puede producir otra ejecución. La migración de índices antiguos adopta solo resultados verificables del mismo proyecto/modelo/idioma, con el mismo estado de voces y sin normalizaciones; en casos ambiguos se conserva la versión anterior y se genera otra.

Los modelos se reutilizan durante un lote y se liberan al terminar. Los audios se copian temporalmente a la máquina de Colab; los originales de Drive se conservan. Los paquetes técnicos de cada ejecución mantienen identificadores distintos, comprobaciones de integridad, texto original, tiempos y controles QA. Los TXT de Lectura representan la última versión generada de cada grabación.

## Alcance de la validación

Los tests automáticos verifican contratos, integridad, selección, guardado, reanudación y comportamiento ante fallos con datos sintéticos. No acreditan precisión de WhisperX, separación de voces ni disponibilidad real de una GPU de Colab. Antes de utilizar citas, revisa el audio y el texto. El glosario normaliza después del reconocimiento; no es vocabulario contextual enviado al modelo.

Colab gratuito no garantiza GPU ni duración. Consulta las [condiciones y límites oficiales](https://research.google.com/colaboratory/faq.html). La ejecución ocurre en Google, y autorizar Drive permite al código del cuaderno acceder a sus archivos. No se publican grabaciones en GitHub por utilizar esta herramienta.

## Cuaderno separado de pruebas

El cuaderno MEIKA_Vox_Colab_Pruebas usa la versión pública y guarda en `MEIKA_Vox_Pruebas/Proyectos`, separado de la carpeta habitual. Evalúa primero un audio corto: comprueba texto, tiempos, etiquetas si corresponden y descarga. Al repetir con el mismo proyecto y configuración debe aparecer como ya existente. No es un certificado de calidad ni una prueba real aprobada.


## Si parece detenido

Durante la preparación verás Cargando librería, Conectando, Descargando… MB o Cargando modelos. La preparación de voces tiene un límite de cinco minutos: si lo supera, se detiene el proceso del motor y aparece un mensaje para reintentar o elegir Solo transcribir. El reconocimiento de un audio largo no usa ese límite. El motor trabaja en un proceso separado del panel para que una librería nativa no bloquee su reloj. Si el proceso termina inesperadamente, el panel informa el fallo y permite recuperar pendientes.

Una sesión que ya está abierta conserva el código cargado. Para recibir una actualización hay que preparar nuevamente el panel, después de detener la tarea activa; actualizar GitHub no modifica una ejecución que ya comenzó.
