# MEIKA Vox

**Transcribe carpetas completas de audio desde tu Google Drive usando Google Colab y WhisperX.**

> El panel público permite elegir una carpeta, transcribir y descargar textos en una misma pantalla.

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeikaLab/meika-vox/blob/main/notebooks/MEIKA_Vox_Colab.ipynb)

Código abierto para procesar entrevistas, reuniones, talleres y grupos focales por lotes. Se ejecuta desde **tu propia cuenta de Google Colab** y guarda transcripciones preliminares y respaldos técnicos en **tu propio Google Drive**. No necesitas instalar Python ni tener cuenta de GitHub.

## Pasos

1. Abre el botón **Abrir en Google Colab** y entra con tu cuenta Google.
2. En **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, elige GPU (T4 si aparece), si está disponible.
3. Pulsa ▶ en **Preparar y abrir MEIKA Vox** y autoriza tu Drive. El panel aparece arriba, en esa misma celda.
4. Elige una carpeta de Mi unidad en la lista: se abre y busca los audios automáticamente.
5. Elige **Solo transcribir · sin token** o **Separar hablantes · con token** y pulsa **Transcribir N audios**. Se usa español y el nombre de la carpeta como proyecto; puedes cambiarlos en el panel.
6. Descarga el ZIP de textos o revisa `Mi unidad/MEIKA_Vox/Proyectos/<nombre_proyecto>/Transcripciones/Lectura`.

[Guía breve: voces, progreso, recuperación y límites](docs/COLAB_USUARIOS.md).

El sistema reanuda el lote y omite los archivos ya completados con el mismo audio y configuración. Usa WhisperX **large-v3** con GPU disponible y **small** en CPU solamente cuando el usuario acepta expresamente ese modo. Los tiempos y la disponibilidad de GPU de Colab son variables.

## Modalidades

| Modalidad | Clave (token) | Estado |
|---|---|---|
| Solo transcribir | No | Implementada: texto y tiempos |
| Separar hablantes con Sherpa-ONNX | No | Integrada, experimental; calidad real por validar |
| Separar hablantes con pyannote | Sí, gratuita de Hugging Face | Implementada; requiere autorización del modelo |

Las tres modalidades usan el mismo modelo de transcripción. No hay una comparación que permita afirmar que un motor de voces es el mejor para tus grabaciones.

[Cuaderno de pruebas del panel](https://colab.research.google.com/github/MeikaLab/meika-vox/blob/feat/user-modes-guide/notebooks/MEIKA_Vox_Colab_Pruebas.ipynb): guarda en `MEIKA_Vox_Pruebas`, separado de los resultados habituales. Su disponibilidad no certifica una prueba real de reconocimiento aprobada.

**Validación:** pruebas automáticas de interfaz, integridad y recuperación aprobadas. La ejecución real del panel en GPU de Colab y la calidad de voces en español siguen pendientes; no se anuncia una validación completa.

## Características

- Reconocimiento de voz y alineación por palabra con WhisperX.
- Separación de hablantes con Sherpa-ONNX sin token (experimental) o pyannote con token.
- Panel guiado: carpeta acotada, selección de audios, idioma y proyecto.
- Progreso por etapas, detención tras el audio actual y reintento de pendientes.
- Descarga ZIP de textos; resultados verificados y versiones técnicas conservadas.
- Modelos reutilizados dentro del lote y liberados al terminar.
- Glosario opcional en JSON aportado por cada usuario (no se aplica uno automáticamente).
- Trazabilidad: SHA-256, metadatos ffprobe, texto original, normalización conservadora, timestamps y QA.
- Resultados TXT y archivos JSON/JSONL; reporte del lote y reanudación de procesos interrumpidos.

Las transcripciones **requieren revisión humana** antes de usar citas como evidencia. MEIKA Vox no realiza por sí solo codificación o análisis cualitativo validado.

## Privacidad y límites

El código está disponible públicamente, pero los audios del usuario **no se suben al repositorio GitHub**. Cada persona autoriza su propio Drive dentro de Colab. Se deben revisar las políticas de Google Colab, Drive y de los proveedores de modelos antes de procesar información sensible.

Colab gratuito **no garantiza GPU, ejecución permanente ni sesiones de duración fija**. Debes mantener conectada la sesión durante el procesamiento. MEIKA Vox es un notebook con controles sencillos, no una aplicación web alojada 24/7.

## Uso local para usuarios técnicos

Requiere Python 3.11+, FFmpeg y ffprobe.

```bash
git clone https://github.com/MeikaLab/meika-vox.git
cd meika-vox
python -m venv .venv
pip install -e ".[whisperx]"
meika-vox doctor --strict
meika-vox transcribe entrevista.m4a --project-id MI_PROYECTO --language es
```

Más información en [Arquitectura](docs/ARCHITECTURE.md) y [Estado actual](docs/CURRENT_STATE.md).

**Licencia Apache-2.0.** **MEIKA LAB** · Desarrollo abierto para investigación social y territorial.
