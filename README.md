# Red neuronal desde cero con NumPy — MNIST

Implementé una red neuronal **completamente a mano**: capa densa, funciones de activación, función de pérdida y *backpropagation* escritos desde cero con `numpy`, sin ningún framework de deep learning. La red clasifica dígitos escritos a mano del dataset MNIST y alcanza **97.4 % de exactitud en test**.

Todo el trabajo está en [`notebooks/mnist.ipynb`](notebooks/mnist.ipynb), organizado en: carga y exploración de datos, implementación capa por capa, entrenamiento con backpropagation manual, evaluación con curvas y matriz de confusión, y conclusiones.

## ¿Por qué funciona backpropagation?

Porque el error no aparece en un solo lugar: es una **consecuencia en cadena**. Un peso mal calibrado en la capa 1 produce activaciones distintas, que producen otras probabilidades, que producen una pérdida alta. Y cuando la causa de algo es una cadena de pasos, la regla de la cadena nos dice exactamente cómo repartir la "culpa" hacia atrás: multiplicar las derivadas locales de cada eslabón.

Backpropagation es eso, organizado: **cada capa recibe la culpa por su salida, y con solo eso calcula la culpa de sus propios pesos y la culpa que le toca a su entrada**, pasándosela a la capa anterior. Ninguna capa necesita saber cómo funcionan las demás; solo necesita su derivada local y el gradiente que le llega.

### La idea, paso a paso con esta red

Cuando la red falla con una imagen, reparto la culpa así, de atrás hacia adelante:

1. **¿Cuánta culpa del error tiene la salida?** Derivando softmax y cross-entropy juntas (las dos cosas que tocan la salida), los términos se cancelan y queda algo casi ridículo de simple: $(P - Y)/N$. Si la red le dio alta probabilidad a una clase equivocada, ese gradiente lo empuja a bajar; si acertó, apenas toca nada.
2. **¿Cuánta culpa tiene la capa 2 (la de softmax)?** Cada peso de esa capa contribuyó al error en proporción a cuánto "empujó" la activación que le llegó. Eso, en matrices, es $A_1^\top (P - Y)$: cuánto se usó cada conexión × cuánta culpa salió por ella. La capa guarda solo lo mínimo necesario (la activación $A_1$ del forward) y su gradiente sale casi gratis.
3. **¿Cuánta culpa tiene la entrada de la capa 2?** Aquí está lo elegante: es la misma cuenta con otro orden, $(P - Y) \, W_2^\top$. Con esto pregunto "¿qué neuronas de la capa oculta empujaron a que la salida se equivocara?" y le paso esa culpa hacia atrás.
4. **¿Cuánta culpa tiene cada neurona de la capa oculta?** Solo si ReLU estaba activa le llega el mensaje (si la neurona estaba apagada, no influyó en nada: su derivada es 0). Ahí muere o fluye, y eso es exactamente la derivada de ReLU.
5. La capa 1 repite el paso 2 con la culpa que recibió: mide cuánto empujó cada píxel por cada conexión y acaba con $X^\top$ multiplicado por la culpa filtrada por ReLU.

Ese estilo de pregunta ("de quién es la culpa por cada salida") se responde una sola vez y **se propaga**: el gradiente de una capa se computa una vez y se reutiliza, en vez de recomputarlo desde cero para cada uno de los ~101 mil pesos (784×128 + 128×10). Por eso backpropagation no es solo *correcto* sino también *barato*: todo el backward cuesta un par de veces el costo del forward, sin importar cuán profunda sea la red.

El desarrollo completo de estas fórmulas, paso a paso y organizado capa por capa, está en el notebook ([`notebooks/mnist.ipynb`](notebooks/mnist.ipynb), sección 3).

## Cómo ejecutar el código

Lo único que hace falta es [`uv`](https://docs.astral.sh/uv/). Desde la raíz del proyecto:

```bash
uv sync
```

Eso crea el entorno virtual `.venv` con las dependencias del proyecto (Python 3.12, numpy, matplotlib, requests, jupyter). Después:

1. Abrir `notebooks/mnist.ipynb` (con `uv run jupyter lab`, Jupyter, o VS Code).
2. Elegir el kernel del `venv` del proyecto (`.venv`).
3. Correr todas las celdas en orden (Run All).

Notas de la primera corrida:

- El dataset **se descarga solo** la primera vez (desde el bucket de TensorFlow/Keras) y queda cacheado en `data/mnist.npz` (~11.5 MB); las corridas siguientes lo leen del disco.
- El entrenamiento completo (10 épocas, mini-batch) tarda **unos minutos** en CPU: es el costo de hacer todos los cálculos a mano en NumPy, sin GPU ni kernels optimizados.
- La semilla global está fijada (`SEED = 42`), así que todas las corridas reproducen los mismos resultados.

## Resumen de lo trabajado

**Arquitectura:** `784 → 128 (ReLU) → 10 (softmax)`. Cada imagen de 28×28 se aplana a un vector de 784 píxeles normalizados a `[0, 1]`; la red termina en una distribución de probabilidad sobre los 10 dígitos.

**Componentes implementados desde cero:**

- **Capa densa** (`Z = X @ W + b`) con inicialización He (`sqrt(2 / n_in)`) para que las activaciones no se apaguen ni exploten, y su `backward` exacto: `dW = Xᵀ dZ`, `db = sum(dZ)`, `dX = dZ Wᵀ`.
- **ReLU** y su derivada binaria: aporta la no linealidad (sin ella apilar capas sería una única transformación lineal) y deja que el gradiente fluya sin encogerse.
- **Softmax** numéricamente estable (resta del máximo por fila antes de exponenciar) para convertir logits en probabilidades.
- **Cross-entropy** con clip para evitar `log(0)`: penaliza fuerte estar seguro y equivocado, y casi nada el acierto seguro.
- **Backpropagation manual** con descenso por gradiente sobre mini-batches de 64 (learning rate 0.1): el punto medio razonable entre ruido y costo por paso.

**Proceso:** Antes de entrenar verifiqué el balance de clases (~6000 ejemplos por dígito, sin sesgo), aplané y normalicé los píxeles, y separé el 10 % del train como validación para medir generalización sin contaminar el aprendizaje. Entrené 10 épocas midiendo pérdida y exactitud sobre train y validación en cada una.

**Resultados (con la semilla fija del notebook):**

| Métrica | Valor |
| --- | --- |
| Exactitud en test | **97.40 %** (9738 de 10000) |
| Pérdida en test | **0.0854** |
| Exactitud en validación | 97.40 % |
| Exactitud en train | 98.54 % |

**Análisis de errores:** la matriz de confusión muestra que el par dominante es **4 → 9** (21 de los 262 errores, más del doble que cualquier otro), seguido por 7→9, 5→3, 7→1 y 2→8. Son confusiones de forma, no de dibujo: comparten silueta a la resolución de 28×28.

**Limitaciones que explora el notebook:** al aplanar la imagen se pierde la estructura espacial (no hay noción de trazo ni de píxeles vecinos, lo que resuelven las redes convolucionales), y la capacidad es limitada (una capa oculta de 128 neuronas distingue los dígitos claros pero no los casos ambiguos). También se observa el estancamiento del final del entrenamiento: la pérdida de validación bajó monótonamente las 10 épocas (0,2397 → 0,0899) y la exactitud de validación tocó su máximo (97,40 %) en las épocas 9 y 10, así que no llegó a aparecer overfitting (la señal clásica sería la pérdida de validación subiendo mientras el train sigue bajando). Lo que sí se ve es el train seguir mejorando mientras la validación deja de mejorar, con una brecha de 1,14 puntos (98,54 % vs 97,40 %): presión *pre*-overfitting, que es justamente el punto donde actuaría un *early stopping* si se entrenaran más épocas.

**Lo que aprendí:** backpropagation no es magia, es la regla de la cadena multiplicando derivadas hacia atrás; softmax + cross-entropy es un regalo matemático cuya derivada conjunta se simplifica a `P - Y`; el ruido de los mini-batches es barato y hasta ayuda a escapar de mínimos malos; y la normalización de entradas junto con la inicialización de pesos deciden entre aprender o no aprender.
