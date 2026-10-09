# Verificación operativa de MEIKA Vox (rama de reparación)

Esta guía valida la transcripción básica de forma gratuita, sin token y sin servicios pagos. **La prueba automática en CPU no sustituye una validación de Colab con GPU.**

## 1. Prueba automatizada y reproducible

El workflow `.github/workflows/engine_smoke.yml` instala WhisperX real, crea un audio sintético en español, transcribe con `tiny` sobre CPU y comprueba que exista un TXT no vacío, un manifiesto y checksums. No mide calidad lingüística, no presupone reconocimiento exacto de palabras y no prueba GPU.

Revisa el estado de las acciones del PR antes de fusionarlo.

## 2. Verificar un audio real en Colab

1. Abre `notebooks/MEIKA_Vox_Colab.ipynb` desde la rama del PR.
2. Selecciona GPU si Colab la ofrece, conecta tu propio Drive y autoriza la instalación.
3. Usa **Solo transcribir**, sin token, con un audio breve y real de 1–3 minutos.
4. Comprueba que aparezcan un TXT con contenido y un `manifest.json` en `Mi unidad/MEIKA_Vox/Proyectos/<PROYECTO>/Transcripciones`.
5. Ejecuta el verificador, cambiando `<PROYECTO>` por el nombre del proyecto:

```python
!python /content/meika-vox/scripts/verify_run.py "/content/drive/MyDrive/MEIKA_Vox/Proyectos/<PROYECTO>/Transcripciones"
```

El verificador exige al menos un manifiesto y que **todos** los resultados encontrados sean íntegros. Los audios cuya exportación falló no se consideran exitosos. `--expect` es opcional para explorar contenido, **no** es una métrica de exactitud.

## 3. Reanudación después de desconexión

Con dos audios, termina el primero y detén la sesión antes de completar el segundo. Vuelve a abrir el cuaderno con la misma configuración. El primer audio debe aparecer como ya existente, sin volver a procesarse. El segundo debe quedar pendiente y transcribirse. El panel nunca marca como finalizado un resultado sin TXT, conteos y checksums completos.

## 4. Diagnóstico de errores

Si falla un audio, comprueba `Detalles técnicos` y `Transcripciones/logs/error_motor_<fecha>.log` en tu Drive. Los informes incluyen etapa, error, traceback del trabajador y una cola de errores nativos. Se ocultan patrones comunes de claves, pero revisa el contenido antes de compartirlo públicamente.

El trabajador puede reiniciarse **para el siguiente audio** después de una caída, con un máximo de tres reinicios por instancia. No vuelve a marcar el audio fallido como exitoso ni lo reintenta sin control. El timeout de una transcripción depende de inactividad de *etapas* y no implica progreso medido por segundos del audio.

## 5. Diarización opcional

Sherpa-ONNX se instala al seleccionar la modalidad sin token. Si la separación de voces falla luego de obtener el texto, se conserva la transcripción y se genera una advertencia `DIARIZATION_FAILED` para revisión humana.

**Criterios aún pendientes antes de declarar el proyecto plenamente validado:** una ejecución completa en Colab con GPU T4, modelo `large-v3`, audio real en español, verificación del ZIP descargado, una recuperación observada y evaluación de voces por separado.
