# Validación del panel de Colab · 9 de octubre de 2026

## Problema reproducido

El cuaderno imprimía el diccionario devuelto por `build_panel`, incluyendo controles y rutas. El desplegable requería entrar explícitamente a la carpeta antes de buscar; esa separación resultaba confusa. Además, `strip()` alteraba nombres reales con espacios finales.

## Correcciones

- Elegir una carpeta abre esa carpeta y busca automáticamente, incluidas sus subcarpetas por defecto.
- Se conservan nombres exactos con espacios y acentos. La ruta manual necesita Abrir ruta.
- Mi unidad no se recorre entera automáticamente. Una carpeta vacía muestra archivos visibles y orientación.
- MP4, WEBM, 3GP y MKV se aceptan como contenedores con audio.
- Una pantalla conserva carpeta, selección, configuración y botón Transcribir N audios. Progreso, textos y reintentos aparecen debajo.
- El cuaderno asigna el panel a una variable para evitar imprimir su representación técnica.

## Verificaciones

La versión pública anterior pasó Ruff y 43 tests en Python 3.11 y 3.12. Los tests ejercitan widgets reales, nombres con espacios finales y Unicode, navegación, búsqueda recursiva y directa, archivos MP4, carpetas vacías, exclusión de grabaciones, consentimiento CPU, guardado, contenido del ZIP, reanudación, fallo, reintento exitoso y activación de la descarga.

El reconocimiento de esos tests usa un proveedor simulado; no mide precisión de voz. El botón de descarga se verifica con un sustituto de la API de Colab.

Se instaló WhisperX 3.8.6 en un entorno CPU aislado y se preparó un audio sintético en español para una prueba real. La revisión automática bloqueó la continuación por una solicitud saliente a telemetría de Microsoft cuyo contenido no se pudo establecer. Esa ejecución no se cuenta como aprobada. No se usaron grabaciones privadas.

Pendiente: transcripción real con large-v3, GPU, montaje de Drive y representación visual interactiva del panel dentro de Colab. Las pruebas de archivos locales no garantizan disponibilidad de GPU, permisos ni estabilidad de una sesión de Google.

## Buscador de carpetas con audios

La búsqueda global es opcional y se ejecuta en un hilo con avance y cancelación. Recorre Mi unidad, evita resultados y rutas externas/cíclicas, y cuenta archivos compatibles directamente en cada carpeta. Finaliza como parcial si se cancela, alcanza 60 segundos/3.000 carpetas o encuentra errores de lectura. Permite elegir una carpeta encontrada o audios sueltos de Mi unidad sin escanear recursivamente toda la unidad al comenzar el lote. Las pruebas verifican límites, cancelación, rutas cíclicas, exclusión, conteos, elección y selección directa en raíz. No demuestra que el montaje de Drive vea todos los archivos de la cuenta.

## Propuesta de modalidades y manual

Modalidades visibles fuera de opciones avanzadas: texto sin token, Sherpa sin token (integrado, experimental), pyannote con token. Solo pyannote muestra la configuración de claves. El cuaderno de pruebas descarga la rama de propuesta y utiliza MEIKA_Vox_Pruebas para resultados separados. Los tests verifican selección explícita del motor Sherpa sin token y que las instrucciones de token se oculten al volver a solo texto. No se repitió la ejecución de reconocimiento bloqueada ni se acredita una prueba en GPU.


## Revisión final de esta propuesta

Sherpa se integra con modelos de publicaciones públicas, descarga atómica y extracción acotada a modelo/licencia/README. Conserva caché, informa progreso de voces y registra motor/modelos en el manifiesto. No requiere ni lee HF_TOKEN. Número exacto de voces opcional; 0 usa agrupamiento automático. Cambiar de motor o cantidad de voces cambia la identidad de reanudación, y no se adoptan resultados antiguos ambiguos de otro motor.

Los segmentos sin hablante conocido ya no se unen en un párrafo gigante. Las pruebas del adaptador usan dobles de la API, sin cargar bibliotecas nativas, descargar modelos ni medir precisión. Esto verifica contratos y protección de archivos, no compatibilidad de ejecución nativa o exactitud de voces en Colab. La prueba real previamente bloqueada no se repitió por otra vía.

Los TXT nuevos muestran tiempos HH:MM:SS y etiquetas Hablante 1, Hablante 2, conservando IDs técnicos en JSON. Los textos ya guardados no se reprocesan por cambios de presentación.


## Corrección de preparación y estado

Las capturas de usuario permanecieron en Preparando separación de voces, sin reloj visible. No permiten concluir si el bloqueo fue de motor o representación de widgets. La revisión identificó que la carga nativa ocurría en el mismo proceso de los widgets y no había límite global de preparación.

El panel ahora utiliza un proceso persistente separado para WhisperX y los motores de voces, con eventos de etapa, detección de salida inesperada y límite de cinco minutos solo para preparación. Conserva modelos entre audios del lote. Los secretos viajan por stdin del proceso local, nunca en argumentos o salidas. Se descarta stdout ajeno al protocolo y no se muestran mensajes originales de excepciones de los proveedores. El reloj aparece desde el inicio; Actualizar estado permite consultar el estado del panel y una espera prolongada se describe sin afirmar avance del reconocimiento. Las descargas informan bytes y validan Content-Length cuando existe.

Las pruebas del transporte ejecutan subprocess reales con un motor sintético sin conexión de red. Comprueban serialización de palabras, etapas, reutilización, terminación inesperada y timeout. No ejecutan WhisperX/Sherpa ni acreditan precisión de audio, compatibilidad nativa o comunicación de widgets en una sesión Colab real.
