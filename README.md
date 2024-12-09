# Sistema di Analisi Biometrica Multimodale

## Descrizione
Sistema avanzato per l'analisi delle espressioni facciali, microespressioni e dati biometrici. Il sistema integra:
- Analisi video in tempo reale
- Rilevamento delle microespressioni
- Tracciamento della dilatazione pupillare
- Integrazione con dispositivi wearable
- Analisi statistica personalizzata
- Calibrazione individuale

## Struttura del Progetto
```
├── src/
│   ├── facial_analysis/      # Moduli per l'analisi facciale
│   ├── pupil_tracking/       # Moduli per il tracciamento pupillare
│   ├── wearable/            # Integrazione dispositivi wearable
│   ├── data_processing/     # Elaborazione e analisi dati
│   └── api/                 # API REST
├── models/                  # Modelli pre-addestrati
├── config/                  # File di configurazione
├── tests/                   # Test unitari
└── docs/                    # Documentazione dettagliata
```

## Installazione
1. Clonare il repository
2. Installare le dipendenze: `pip install -r requirements.txt`
3. Scaricare i modelli pre-addestrati necessari

## Utilizzo
1. Configurare il file `.env` con le impostazioni personali
2. Eseguire la calibrazione iniziale
3. Avviare il sistema: `python src/main.py`

## Tecnologie Utilizzate
- MediaPipe per il rilevamento dei punti facciali
- OpenCV per l'elaborazione video
- dlib per il tracciamento facciale avanzato
- pandas per la gestione dei dati
- scikit-learn per l'analisi statistica
- Flask per l'API di servizio
