Cats Vs Dogs classifier è una applicazione python realizzata in VSCode che fa uso di una semplice Convolutional Neural Network
progettata per verificare se in una immagine sia presente un cane o un gatto (classificazione binaria).

L'applicazione è composta da da due parti distinte:

1) costruzione, training e salvataggio della rete

La CNN è stata progettata, implementata e testata utilizzando il framework Tensorflow e le API di Keras. 
L'implementazione si trova nel file "cats_vs_dogsCNN.ipynb" in formato Jupiter Notebook. 

Per il training della CNN è stato utilizzato il dataset pubblico "cats_vs_dogs" composto da 24.998 immagini di cani e gatti 
equamente distribuite. Di queste immagini 20.000 vengono usate per il training della rete (train_data) mentre le rimanenti
vengono utilizzate per verificare l'accuratezza della rete durante la fase di training (test_data).

Per velocizzare il training si è scelto di scalare le foto a 160 x 160 pixel. Se si utilizzassero foto di dimensioni maggiori si 
otterrebbero risultati ancora migliori ma la fase di training richiederebbe un tempo molto più lungo a causa del numero
di parametri della rete.

Una volta terminata la fase di training l'architettura della CNN, ed i suoi pesi, vengono salvate nel file "cats_dogs_cnn_model.h5"
in modo da poter essere utilizzata successivamente dall'applicazione per la classificazione.

Di seguito l'architettura della rete:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 158, 158, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 79, 79, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 77, 77, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 38, 38, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 36, 36, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 18, 18, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 16, 16, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 8, 8, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 8192)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │     4,194,816 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           513 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

I parametri utilizzati nella fase di training della CNN sono: 

optimizer = Adam (backpropagation)
learning rate = 0.0001
loss = binary_crossentropy
metric = accuracy
batch size = 32 (default)
epoch = 9

2) UI per per eseguire la classificazione binaria cane/gatto

La UI (file "cats_vs_dogs_classifier.py") è stata realizzata utilizzando tkinter e customtkinter, questa presenta una interfaccia 
grafica che permette di selezionare una immagine da disco che successivamente viene passata alla CNN per eseguire la predict. 
Il risultato della predict viene riportato, insieme al grado di confidenza della rete, nel pannello di sinistra della 
applicazione, mentre sul pannello di destra viene mostrata la foto selezionata dall'utente.

Essendo la rete costruita per prendere in input immagini 160 x 160 la foto viene scalata a tali dimensioni prima di essere 
passata alla predict.

Nella directory test_images sono contenute alcune immagini di cani e gatti scaricate da internet e che la rete non ha mai
visto durante la fase di training.


#### Installazione della applicazione su MAC con chip ARM o Intel 

requisiti: VSCode, python v 3.10 o superiore

1) copiare il programma in una directory
2) posizionarsi all'interno della directory con terminal
3) da terminal creare un ambiente virtuale: python -m venv .venv
4) da terminal attivare l'ambiente virtuale appena creato: source .venv/bin/activate
5) installare tutte le dipendenze: pip install -r requirements.txt
6) lanciare l'applicazione: python cats_vs_dogs_classifier.py

Se si desidera aprire il file "cats_vs_dogsCNN.ipynb" è necessario lanciare VSCode, aprire la directory da VSCode e poi 
aprire il file.