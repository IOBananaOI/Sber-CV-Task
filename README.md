# Sber-CV-Task

Данный репозиторий представляет собой тестовое задание на позицию CV Engineer в СБЕР.

# Оглавление
- [Устройство VLM](#устройство_vlm)
- [Выбор датасета](#выбор_датасета)
- [Выбор модели](#выбор_модели)
- [Выбор метрик](#выбор_метрик)
- [Дообучение модели](#дообучение)
- [Результаты](#результаты)
- [Использование модели](#usage)
- [Итоги](#results)


# Устройство VLM
<a name="устройство_vlm"></a>

Визуальные языковые модели (VLM, Visual Language Models) представляют собой комбинацию двух основных типов нейронных сетей: моделей обработки естественного языка (NLP) и моделей обработки изображений (CV). Эти модели предназначены для объединения текстовой и визуальной информации, что позволяет им решать задачи, требующие понимания как текста, так и изображений. Примерами таких задач являются вVisual Question Answering, Image Captioning, Chart Question Answering и др.

### Архитектура VLM

Основные компоненты архитектуры VLM включают:

1. **Image-Encoder:**
   - **Сверточные нейронные сети (CNN):** Традиционно, экстракция признаков из изображений осуществлялась с помощью сверточных нейронных сетей, таких как ResNet, VGG, или Inception. Эти сети извлекают пространственные признаки, такие как контуры, текстуры и более сложные визуальные объекты.
   - **Vision Transformers (ViT):** В последнее время использование трансформеров для обработки изображений стало популярным. Эти модели, такие как ViT, работают с изображениями, разделенными на патчи, и обучаются аналогично NLP моделям на основе трансформеров.

2. **Text-Encoder:**
   - **Трансформеры:** Для обработки текстовых данных используется архитектура трансформеров, таких как BERT, GPT, T5 и их вариации. Эти модели хорошо захватывают семантические и синтаксические зависимости в тексте.

3. **Multimodal Integrator:**
   Эта часть архитектуры отвечает за интеграцию визуальной и текстовой информации. Обычно для этой цели используются многомодальные трансформеры, которые могут обрабатывать текстовые и визуальные признаки одновременно.
.

4. **Decoder:**
   - Для генерации окончательного ответа используется, как правило, ещё один трансформер, который принимает на вход объединенные визуальные и текстовые признаки, и генерирует текст.

### Обучение VLM

Процесс VLM состоит из нескольких этапов:

1. **Подготовка данных:**
   - **Аннотированные датасеты:** Требуются большие аннотированные датасеты, содержащие как изображения, так и связанные с ними тексты (например, COCO, Visual Genome, или концептуально составленные датасеты).
   - **Предобучение (pre-training):** Модели часто проходят стадию преобучения на больших объемах данных без аннотаций.

2. **Обучение на конкретной задаче (fine-tuning):**
   - **Обучение с учителем (Supervised Learning):** На основе аннотированных данных модель обучается предсказывать конкретный выход (например, подпись к изображению).
   - **Обучение с подкреплением (Reinforcement Learning):** Иногда используется для задач, где нужно учитывать последовательные действия, например, диалоговые системы с визуальной поддержкой.

# Выбор датасета
<a name="выбор_датасета"></a> 
В качестве датасета был выбран <a href="https://arxiv.org/pdf/2308.01979">RealCQA</a>, по той причине, что наиболее используемые бенчмарки содержат в себе либо синтетические данные и синтетические графики (FigureQA, DVQA), либо графики, построенные на реальных данных, но при этом сами созданные искусственно (PlotQA, ChartQA, Leaf-QA). Ни одна из данных двух групп не справляется со сложностью распределения реальных диаграмм, описанных в, например, таких областях данных, как научная литература. RealCQA в свою очередь состоит из реальных диаграмм, извлеченных из научных работ. Таким образом, представляет интерес дообучение и проверка возможностей CQA моделей на данном датасете.


<img src="https://github.com/user-attachments/assets/df287b2f-ff15-455c-b896-33e3281d35d8" width="800"/>

*Изображение взято из оригинальной статьи <a href="https://arxiv.org/pdf/2308.01979">RealCQA</a>*

В качестве предобработки все лейблы датасета, представляющие собой списки были преобразованы в строки. Также json файлы всех картинок были слиты в один.


# Выбор модели
<a name="выбор_модели"></a>
В качестве основной модели была выбрана <a href="https://arxiv.org/pdf/2212.09662">MatCha</a>. Данная модель по архитектуре представляет собой Pix2Struct, предобученный на 2 задачи: math reasoning и chart
derendering. Как показали авторы статьи, это дало существенный прирост по метрикам, вследствие чего модель на момент выхода была SOTA. 

![image](https://github.com/user-attachments/assets/919475ce-6a57-4216-97fd-94f0d35e6b66)


Мною данная модель была выбрана по **нескольким причинам**: 
- она достаточно маленькая (~ 282M параметров), что позволяет легко дообучить её, имея небольшие вычислительные мощности;
- несмотря на первый пункт, данная модель показывает хороший перфоманс, например, <a href="https://paperswithcode.com/sota/chart-question-answering-on-plotqa">согласно PaperWithCode</a> MatCha является SOTA на датасете PlotQA до сих пор;
- разработчики предоставляют возможности удобного использования и дообучения модели посредством HuggingFace;

Данная модель не обучалась на выбранном мною датасете RealCQA, так что дообучение имеет смысл.

Помимо прочего, на HuggingFace есть несколько подходящих нам моделей MatCha, а именно оригинальная (matcha-base), дообученная на PlotQA (matcha-plotqa) и дообученная на ChartQA (matcha-chartqa). Я взял все три модели, чтобы проверить, какая из них покажет себя на выбранном датасете лучше всего.



# Выбор метрик
<a name="выбор_метрик"></a>

В качестве метрик мною были выбраны Exact match, Relaxed correctness с порогом 5% (для лучшего оценивания математических вопросов) и average normalized Levenshtein similarity (ANLS), как общеупотребительные метрики для данной задачи.

# Дообучение модели
<a name="дообучение"></a>
Перед началом дообучения был взят исходный датасет и json файлы из него были объединены в один для обеспечения лучшей эффективности во время обучения и инференса. Ссылку на скачивание данного файла можно найти в разделе **Использование модели**.

В качестве функции потерь используется встроенная в модель Huggingface функция кросс-энтропии. Все модели обучались с одинаковыми гиперпараметрами, с ограничением в 600 шагов, поскольку дальнейшее обучение приводило к заметному ухудшению метрик. После каждого 100-го шага производилась валидация моделей, на подмножестве размером ~4к экземпляров, результаты которой отслеживались (см. раздел Результаты). Во время сэмплирования троек вопрос-ответ-картинка, пара вопрос-ответ выбирается случайно, поскольку у многих картинок 3-4 таких пар. Это обеспечивает большее разнообразие данных, нежели в случае выбора какого-то конкретного вопроса. Так, мы можем полагать, присутствие каждого вопроса в трейновой выборке.

На протяжении экспериментов использовалась следующая спецификация гиперпараметров:
- Batch_size: 4;
- Оптимизатор AdamW;
- Косинусный lr scheduler c warmup в 120 шагов, num_cycles=0.5, и initial lr = 1e-4;
- Random ceed = 42; 

Код с обучением можно найти в файле **main.ipynb**.

Помимо прочего, были протестированы другие оптимизаторы, например Adafactor, который рекомендуется для дообучения на странице <a href="https://huggingface.co/docs/transformers/model_doc/matcha">MatCha Hugging Face</a>, однако в моём случае AdamW оказался более предпочтительным в силу лучшей сходимости.

# Результаты
<a name="результаты"></a>

Перейдём к описанию полученных результатов. Для начала взглянем на график лосса на обучающей выборке. Мною был взят каждый 15-й шаг с той целью, чтобы график был более удобно читаем, поскольку мелкие колебания и точные значения на каждом шаге интересуют нас меньше, чем видение общей динамики обучения.
![image](https://github.com/user-attachments/assets/e3c21e38-16b9-4e0f-8031-0362d5033f53)

Как можно видеть, все три модели достаточно неплохо сходятся и в целом имеют одинаковые результаты на трейновой выборке.

Перейдем к лоссу на тестовой выборке.

![image](https://github.com/user-attachments/assets/de10659d-818c-4715-9f9b-79e381764f48)

Здесь можно видеть, что matcha-base немного обходит своих соперников, далее идёт matcha-chartqa и затем matcha-plotqa.

Перейдём к более содержательным метрикам.

Начнём с ANLS, как можно видеть, тенденция сохраняется и matcha-base здесь уже заметно превалирует над остальными моделями.\

![image](https://github.com/user-attachments/assets/4bf2ffd6-97c0-481f-acb0-bb63c57390b3)

Теперь рассмотрим Exact match.
![image](https://github.com/user-attachments/assets/ec7ea427-1e21-4b8a-b82f-4d211542cae9)


И в конце посмотрим на график Relaxed correctness.

![image](https://github.com/user-attachments/assets/4d1a7f2e-8125-4059-acb6-2a609e210ca7)


По итогам графиков можно видеть, что matcha-base обошла своих конкурентов по всем релевантным метрикам.

Рассмотрим полученные результаты более конкретно.

## Метрики до обучения

Перед обучением все три модели были протестированы на той же самой отложенной выборке, что и впоследствие во время обучения. Результаты получились следующие.

| Model | EM   | ANLS   | Relaxed correctness   |
|----------------|--------|--------|------------|
| matcha-base   | **0.0** | **0.0** | **0.0** |
| matcha-chartqa   | **13.13%** | **13.25%** | **13.51%** |
| matcha-plotqa   | **17.38%** | **17.71%** | **17.84%** |

## Метрики после обучения

После обучения лучшие метрики вышли такие. Взяты результаты моделей при валидации на 500 шаге для matcha-base и matcha-chartqa и на 600 шаге для matcha-plotqa, поскольку именно там модели достигают наивысших метрик.

| Model | EM   | ANLS   | Relaxed correctness   |
|----------------|--------|--------|------------|
| matcha-base   | **80.8%** | **82.37%** | **80.91%** |
| matcha-chartqa   | **76.67%** | **78.63%** | **76.88%**|
| matcha-plotqa   | **71.93%** | **75.46%** | **72.09%** |

Как можно видеть matcha-base обходит matcha-chartqa в среднем на 3.96% по всем метрикам и matcha-plotqa на 8.2%. Сама по себе модель показывает достаточно неплохие результаты, если учитывать её размер и сложность датасета.

Кроме того видна интересная тенденция, чем лучше показала себя модель на тестовой выборке относительно других моделей до обучения, тем хуже она проявила себя после.

Изучив сэмплы, на которых модель ошибалась больше всего, я выяснил, что эти задачи включали в себя, как правило, вопросы на вычисления, то есть хуже всего модель справляется с математическими задачами, в свою очередь на бинарные вопросах, вопросах на определение типа графика, перечисления чего-то модель практически не ошибается. Возможно, для устранения этой проблемы в будущем, стоит увеличить долю математических вопросов в датасете.

Помимо прочего, мною была использована BitsAndBytes квантизация итоговой модели в int8, однако желаемых результатов она не дала, возможно следует попробовать снова с более тонкой настройкой квантизируемых слоёв.

## Пример работы модели

![image](https://github.com/user-attachments/assets/c230a1dd-5219-450c-90a5-f054a5282e66)

```
Question: "What is the type of chart?"
True answer: "Line chart"
Predicted answer: "Line chart"
```
*(см. в main.ipynb)*

# Использование модели
<a name="usage"></a>

Веса итоговой модели matcha-base могут быть получены по <a href="https://drive.google.com/file/d/1UTPk8yM2_o2Cm8DBmhmmKDozpV8TBvk7/view?usp=sharing">данной ссылке</a>. В случае желания протестировать модель самостоятельно, можно воспользоваться ноутбуком validation.ipynb, однако заранее также убедитесь в наличии файла <a href="https://drive.google.com/file/d/1VMsPdd3znZaHhvoYx5QZuSIQQ6hPMjo6/view?usp=sharing">merged_qa.json</a> в директории data.


# Итоги
<a name="results"></a>

По итогам данной работы, можно сделать вывод, что MatCha - интересная и мощная модель для своего размера, которая может быть легко дообучена для различных задач. Однако, что оказалось лично для меня неожиданным, базовая модель обучилась на новом датасете гораздо лучше, нежели её сородичи, предобученные на ChartQA и PlotQA. Возможно это связано с тем, что данные датасеты сильно отличаются по парам вопрос-ответ, из-за чего данные предобученные модели сложнее справлялись с данными из другого датасета. Тем не менее, можно заметить что предобученные модели, перед дообучением на RealCQA справились с этим датасетом всё таки лучше, чем matcha-base, это опять же связано со спецификой предобучения оригинальной MatCha, ведь предобучалась она на в первую очередь Chart derendering и Math reasoning, что можно видеть в файле предсказаний недообученной модели (см. results/matcha-base/predictions_before_training.csv).

Стоит отметить, что было бы интересно также рассмотреть и другие похожие по размерам модели, например ближайшего конкурента UniChart, а также DePlot с какой-нибудь LLM в паре.

Также неплохо было бы квантизировать эту модель должным способом, дабы увеличить её эффективность на инференсе.
