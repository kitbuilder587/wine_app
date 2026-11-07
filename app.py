import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent
LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_batch(content, model, features, threshold):
    frame = pd.read_csv(BytesIO(content))
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В файле нет строк с винами')
    values = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Заполните пропуски и замените бесконечные значения числами')
    if not values['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 или 1')
    result = frame.copy()
    result['probability'] = model.predict_proba(values)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


def main():
    st.set_page_config(page_title='Подбор вина', layout='wide')
    st.title('Подбор вина')
    st.caption('Лабораторные показатели помогают выбрать вина с оценкой экспертов не ниже 7.')
    try:
        model, data, config = load_model(), load_data(), load_config()
    except (OSError, ValueError, ImportError) as error:
        st.error(f'Не удалось открыть файлы приложения. {error}')
        st.stop()
    features = config['features']
    X_test, y_test = data.loc[:, features], data['target']
    probability = test_proba(model, X_test)
    threshold = st.sidebar.slider('Порог отбора', min_value=0.0, max_value=1.0,
                                  value=float(config['threshold']), step=0.001,
                                  format='%.4f', key='threshold')
    st.sidebar.caption(f'Исходный порог {config["threshold"]:.4f}. Он выбран на валидации.')
    st.sidebar.write('Один порог применяется ко всем вкладкам.')
    one, evaluation, batch, weights = st.tabs([
        'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

    with one:
        st.subheader('Показатели вина')
        st.caption('Начальные значения равны медианам тестовых данных. Диапазоны взяты из той же таблицы.')
        inputs = {}
        columns = st.columns(3)
        for index, name in enumerate(features):
            with columns[index % 3]:
                if name == 'is_red':
                    inputs[name] = int(st.toggle('Красное вино', value=bool(X_test[name].median())))
                else:
                    inputs[name] = st.number_input(LABELS[name],
                        min_value=float(X_test[name].min()), max_value=float(X_test[name].max()),
                        value=float(X_test[name].median()), format='%.4f', key=name)
        sample = pd.DataFrame([inputs], columns=features)
        score = float(model.predict_proba(sample)[0, 1])
        st.metric('Вероятность хорошего вина', f'{score:.1%}')
        if score >= threshold:
            st.success('Модель рекомендует включить вино в подборку')
        else:
            st.info('Модель не включает вино в подборку при этом пороге')

    with evaluation:
        st.subheader('Качество на тестовой выборке')
        prediction = (probability >= threshold).astype(int)
        precision = metrics.precision_score(y_test, prediction, zero_division=0)
        recall = metrics.recall_score(y_test, prediction, zero_division=0)
        f1 = metrics.f1_score(y_test, prediction, zero_division=0)
        columns = st.columns(3)
        columns[0].metric('Precision', f'{precision:.3f}')
        columns[1].metric('Recall', f'{recall:.3f}')
        columns[2].metric('F1', f'{f1:.3f}')
        st.write('Более высокий порог сокращает подборку и уменьшает число ошибочных рекомендаций. '
                 'При снижении порога в подборку попадает больше хороших вин, но вместе с ними могут попасть посредственные.')
        st.caption('Precision показывает долю хороших вин среди выбранных. Recall показывает долю найденных хороших вин. '
                   'При пустой подборке precision и F1 принимаются равными нулю.')
        cm = metrics.confusion_matrix(y_test, prediction, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set(xlabel='Решение модели', ylabel='Оценка экспертов')
        st.pyplot(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fpr, tpr, _ = metrics.roc_curve(y_test, probability)
        pr, rc, _ = metrics.precision_recall_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=f'ROC-AUC {metrics.roc_auc_score(y_test, probability):.3f}')
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        tn, fp, fn, tp = cm.ravel()
        axes[0].scatter(fp / (fp + tn), recall, color='darkred', label='Текущий порог', zorder=5)
        axes[1].plot(rc, pr, label=f'PR-AUC {metrics.average_precision_score(y_test, probability):.3f}')
        axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайный отбор')
        axes[1].scatter(recall, precision, color='darkred', label='Текущий порог', zorder=5)
        axes[0].set(xlabel='False positive rate', ylabel='Recall', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[1].set(xlabel='Recall', ylabel='Precision', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        for axis in axes:
            axis.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if not prediction.any():
            st.caption('В подборке нет вин. На PR-графике показан recall 0 и условный precision 0.')

    with batch:
        st.subheader('Оценка файла с винами')
        st.write('Загрузите CSV с 12 признаками. Столбец target не нужен. Порядок столбцов может быть любым.')
        st.download_button('Скачать пример CSV', data=(APP_DIR / 'example.csv').read_bytes(),
                           file_name='example.csv', mime='text/csv')
        uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
        if uploaded is not None:
            try:
                result = predict_batch(uploaded.getvalue(), model, features, threshold)
            except (ValueError, UnicodeError, pd.errors.ParserError) as error:
                st.error(f'Не удалось обработать CSV. {error}')
            else:
                st.dataframe(result, hide_index=True)
                st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                                   file_name='wine_predictions.csv', mime='text/csv')

    with weights:
        st.subheader('Вклад признаков')
        estimator = model.steps[-1][1]
        coefficients = pd.Series(estimator.coef_[0], index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([LABELS[name] for name in coefficients.index], coefficients,
                color=['#9b4762' if value < 0 else '#267c74' for value in coefficients])
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('Коэффициент после масштабирования')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        positive = coefficients[coefficients > 0].sort_values(ascending=False)
        negative = coefficients[coefficients < 0]
        st.write('Сильнее всего за хорошее вино голосует признак ' + LABELS[positive.index[0]].lower() + '.')
        st.write('Сильнее всего против голосует признак ' + LABELS[negative.index[0]].lower() + '.')
        st.caption('Коэффициенты относятся к стандартизованным признакам и учитывают остальные показатели. '
                   'Это связи в данных, а не доказательство причинного влияния.')


if __name__ == '__main__':
    main()
