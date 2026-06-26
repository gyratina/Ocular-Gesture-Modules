###
# project: Ocular Gestures Module (OGM)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# start: 2026-06-26 (yyyy-mm-dd)

import cv2
import time
import argparse
from blinkModule import BlinkDetector


def on_blink_detected(count):
    print(f"\n[OGM] Rilevato battito di ciglia! Totale: {count}")


def main():
    # Gestione degli argomenti da riga di comando per massima flessibilità (es. in Distrobox)
    parser = argparse.ArgumentParser(description="Ocular Gestures Module (OGM) - Test Client")
    parser.add_argument(
        "--headless", "-hl", 
        action="store_true", 
        help="Avvia in modalità headless (senza interfaccia grafica cv2.imshow)"
    )
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=0, 
        help="Indice della fotocamera/webcam da utilizzare (default: 0)"
    )
    parser.add_argument(
        "--threshold", "-t", 
        type=float, 
        default=0.20, 
        help="Soglia matematica EAR (default: 0.20)"
    )
    parser.add_argument(
        "--frames", "-f", 
        type=int, 
        default=3, 
        help="Numero di frame consecutivi ad occhio chiuso per registrare un battito (default: 3)"
    )
    args = parser.parse_args()

    # Inizializza il rilevatore di battiti
    detector = BlinkDetector(ear_threshold=args.threshold, k_frame_threshold=args.frames)
    detector.on_blink_callback = on_blink_detected

    print(f"Inizializzazione della fotocamera (webcam {args.camera})...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Errore: Impossibile aprire la fotocamera {args.camera}.")
        print("Verifica che il dispositivo sia connesso ed esposto all'interno di Distrobox.")
        print("Tip: Verifica i permessi di /dev/video* nel container.")
        detector.close()
        return

    if args.headless:
        print("Avvio in modalità HEADLESS. Premi Ctrl+C nel terminale per uscire.")
    else:
        print("Avvio in modalità GRAFICA. Premi 'q' o 'ESC' sulla finestra video per uscire.")

    try:
        last_print_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nella lettura del frame dalla webcam.")
                break

            # Ribalta il frame orizzontalmente per un effetto specchio naturale
            frame = cv2.flip(frame, 1)

            # Elabora il frame per calcolare l'EAR e rilevare i battiti
            ear, blink_count = detector.process_frame(frame)

            if args.headless:
                # In modalità headless stampiamo periodicamente lo stato a riga di comando
                current_time = time.time()
                if current_time - last_print_time >= 0.2:  # Limita l'output a 5 volte al secondo
                    status_text = "Occhi Chiusi" if (ear is not None and ear < detector.EAR_THRESHOLD) else "Occhi Aperti"
                    ear_val = f"{ear:.3f}" if ear is not None else "N/D"
                    print(f"\rEAR: {ear_val} | Stato: {status_text} | Battiti totali: {blink_count}", end="", flush=True)
                    last_print_time = current_time
                
                # Un piccolo delay per non saturare la CPU in assenza di waitKey
                time.sleep(0.01)
            else:
                # Overlay grafico
                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if ear is not None:
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    
                    if ear < detector.EAR_THRESHOLD:
                        cv2.putText(
                            frame,
                            "OCCHI CHIUSI",
                            (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.putText(
                        frame,
                        "Viso non rilevato",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Mostra la finestra video
                cv2.imshow("Ocular Gestures Module (OGM) - Test", frame)

                # Gestione uscita tastiera
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\nInterruzione da tastiera ricevuta.")
    finally:
        # Rilascia le risorse
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        detector.close()
        print("\nSessione terminata con successo.")


if __name__ == "__main__":
    main()
