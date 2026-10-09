# MEIKA Vox

**Transcribe carpetas completas de audio desde tu Google Drive usando Google Colab y WhisperX.**

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeikaLab/meika-vox/blob/main/notebooks/MEIKA_Vox_Colab.ipynb)

Código abierto para procesar entrevistas, reuniones, talleres y grupos focales por lotes. Se ejecuta desde **tu propia cuenta de Google Colab** y guarda transcripciones preliminares y respaldos técnicos en **tu propio Google Drive**. No necesitas instalar Python ni tener cuenta de GitHub.

## Pasos

1. Abre el botón **Abrir en Google Colab** y entra con tu cuenta Google.
2. En **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, elige GPU si está disponible.
3. Pulsa **Ejecutar todas** y autoriza tu Google Drive.
4. En el panel elige una carpeta específica de Mi unidad y pulsa **Buscar audios aquí**.
5. Confirma proyecto, idioma y audios. Pulsa **Transcribir seleccionados**.
6. Descarga el ZIP de textos o revisa `Mi unidad/MEIKA_Vox/Proyectos/<nombre_proyecto>/Transcripciones/Lectura`.

[Guía breve: voces, progreso, recuperación y límites](docs/COLAB_USUARIOS.md).

El sistema reanuda el lote y omite los archivos ya completados con el mismo audio y configuración. Usa WhisperX **large-v3** con GPU disponible y **small** en CPU solamente cuando el usuario acepta expresamente ese modo. Los tiempos y la disponibilidad de GPU de Colab son variables.

## Características

- Reconocimiento de voz y alineación por palabra con WhisperX.
- Separación de hablantes opcional con pyannote (requiere token y autorización).
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
