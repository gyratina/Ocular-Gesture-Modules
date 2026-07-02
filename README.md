# Ocular Gesture Modules (OGM)

OGM è una libreria Python "turn-key" (chiavi in mano) per il rilevamento in tempo reale dei battiti di ciglia e delle gesture oculari (es. doppio battito, occhiolino).
Sfrutta MediaPipe Face Mesh e OpenCV per estrarre l'EAR (Eye Aspect Ratio) ed esporre un'interfaccia a Macchina a Stati per combinare azioni complesse.

## Installazione

Puoi installare le dipendenze direttamente tramite `pip`:
```bash
pip install .
```

## Utilizzo

```python
from ogm import BlinkDetector, CameraConfig

detector = BlinkDetector()
detector.start(mode="detect")
```
