### 1. Requisiti di Sistema e Dipendenze

Per far girare questo modulo, tu (e chiunque importerà la tua libreria) dovrete avere un ambiente Python configurato con queste tre librerie fondamentali. L'installazione si fa con un banale:
`pip install opencv-python mediapipe scipy`
`python main.py -c 1`

Ecco perché le usiamo e non possiamo farne a meno:

* **`opencv-python` (OpenCV):** * *Motivazione:* È lo standard industriale per la gestione dei flussi video. Ci serve per catturare i fotogrammi dalla webcam e, in fase di test, per disegnare a schermo i risultati. Il tuo modulo riceverà in pasto i frame generati da questa libreria.
* **`mediapipe` (Google Face Mesh):**
* *Motivazione:* È il nostro "estrattore di dati grezzi". Invece di addestrare una rete neurale da zero, usiamo Face Mesh perché ci garantisce il tracciamento di 478 punti facciali in tempo reale con un impatto sulla CPU quasi nullo. È l'unica libreria di IA che ci serve.


* **`scipy` (Scientific Python):**
* *Motivazione:* Ci serve specificamente per il sottomodulo `scipy.spatial.distance`. Per calcolare se un occhio è aperto o chiuso, dobbiamo misurare continuamente la distanza in pixel tra vari punti. Potremmo scriverci noi la formula in Python puro, ma la funzione `euclidean` di SciPy è ottimizzata in C sotto il cofano ed è infinitamente più veloce, annullando la latenza.



---

### 2. Il Motore Matematico: Cosa succede sotto il cofano

Il rilevamento dei battiti di ciglia non si basa su un'intelligenza artificiale che "capisce" cosa sia un occhio chiuso, ma su un calcolo geometrico deterministico. Ecco i tre passaggi matematici che il tuo modulo deve eseguire per ogni singolo fotogramma.

#### Fase A: Estrazione e Normalizzazione delle Coordinate

MediaPipe analizza il viso e restituisce i punti di repere (landmarks). Per un singolo occhio, estraiamo 6 punti specifici che ne definiscono il contorno. Chiameremo questi punti da $P_1$ a $P_6$:

* $P_1$: Angolo esterno dell'occhio.
* $P_2, P_3$: Bordo della palpebra superiore.
* $P_4$: Angolo interno dell'occhio.
* $P_5, P_6$: Bordo della palpebra inferiore.

MediaPipe restituisce queste coordinate come valori normalizzati compresi tra **0.0 e 1.0**. Per usarle matematicamente, dobbiamo convertirle in pixel reali sullo schermo moltiplicandole per la larghezza ($W$) e l'altezza ($H$) del fotogramma video:

$$x_{pixel} = x_{norm} \cdot W$$

$$y_{pixel} = y_{norm} \cdot H$$

#### Fase B: La Distanza Euclidea

Una volta ottenuti i pixel esatti, calcoliamo la distanza fisica tra i punti opposti dell'occhio. Per farlo, usiamo il teorema di Pitagora applicato al piano cartesiano, ovvero la **Distanza Euclidea**.
Se abbiamo due punti $P(x_1, y_1)$ e $Q(x_2, y_2)$, la distanza $d$ tra loro è:

$$d(P, Q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

Il nostro modulo usa SciPy per calcolare:

1. **L'altezza dell'occhio:** La distanza tra i punti verticali $P_2-P_6$ e $P_3-P_5$.
2. **La larghezza dell'occhio:** La distanza tra i punti orizzontali $P_1-P_4$.

#### Fase C: L'Equazione EAR (Eye Aspect Ratio)

Qui entra in gioco la vera magia teorizzata nel 2016 da Tereza Soukupova. Il problema di usare i pixel nudi e crudi è che se l'utente si allontana dalla webcam, l'occhio diventa più piccolo e le distanze in pixel diminuiscono, ingannando il sistema.

Per risolvere il problema, calcoliamo l'**EAR**, un rapporto adimensionale (senza unità di misura) che mette in relazione altezza e larghezza:

$$EAR = \frac{d(P_2, P_6) + d(P_3, P_5)}{2 \cdot d(P_1, P_4)}$$

* **Principio:** Al numeratore sommiamo le due altezze dell'occhio. Al denominatore mettiamo la larghezza moltiplicata per 2 (per bilanciare la somma al numeratore).
* **Perché funziona:** Poiché è una divisione, il risultato rimane costante sia che l'utente stia a 10 centimetri dalla webcam, sia che stia a 2 metri. Se l'occhio è aperto, l'EAR oscilla tra **0.25 e 0.35**. Quando le palpebre si chiudono, il numeratore tende a zero e l'EAR crolla istantaneamente sotto lo **0.10**.

---

### 3. La Logica Temporale (Il Filtro Anti-Rumore)

Calcolare la matematica non basta. Se la webcam perde un fotogramma, l'EAR potrebbe crollare a zero per un millesimo di secondo, generando un "falso positivo" (il tuo modulo direbbe all'app che l'utente ha sbattuto le ciglia anche se non è vero).

Per rendere il tuo modulo *production-ready*, devi implementare una **Macchina a Stati Temporale**.

Devi definire due costanti nel tuo costruttore:

1. `EAR_THRESHOLD`: La soglia matematica (es. **0.20**). Sotto questo valore, consideriamo l'occhio "potenzialmente" chiuso.
2. `CONSECUTIVE_FRAMES`: Il numero di fotogrammi (es. **3**). L'occhio deve rimanere sotto la soglia per almeno 3 frame di fila per essere registrato come battito effettivo.

**Algoritmo logico di esecuzione:**

1. Il frame entra e calcoli l'EAR.
2. Se $EAR < EAR_{THRESHOLD}$:
* Incrementi un contatore (`blink_counter += 1`).


3. Se $EAR \ge EAR_{THRESHOLD}$:
* Controlli il contatore. Se `blink_counter >= CONSECUTIVE_FRAMES`, significa che l'occhio si era chiuso e ora si è riaperto. **Questo è un battito di ciglia completo.** Lancia l'evento allo sviluppatore (`on_blink()`).
* Azzera il contatore (`blink_counter = 0`).
