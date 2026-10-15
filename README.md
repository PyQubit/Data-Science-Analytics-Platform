<div align="center">

<a href="https://github.com/PyQubit">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=28&duration=3000&pause=900&color=2F80ED&center=true&vCenter=true&width=760&lines=Data+Science+Analytics+Platform;Explore+%E2%80%A2+Analyze+%E2%80%A2+Discover+Insights" alt="Data Science Analytics Platform" />
</a>

### Turn raw datasets into transparent, evidence-backed insights.
### تبدیل داده‌های خام به بینش‌های شفاف و مبتنی بر شواهد

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passed-2ea44f?style=for-the-badge&logo=pytest&logoColor=white)](BUILD_STATUS.md)
[![Local First](https://img.shields.io/badge/Analytics-Local--First-5B6DCD?style=for-the-badge)](#-design-principles)

[English](#english) · [فارسی](#فارسی) · [Repository](https://github.com/PyQubit) · [Developer Portfolio](https://pyqubit.github.io/)

</div>

---

<a id="english"></a>
# 🇬🇧 English

## Overview

**Data Science Analytics Platform** is a modular, local-first workbench for exploring, validating, cleaning, analyzing, and reporting on datasets through an interactive Streamlit interface. It brings data quality checks, exploratory analysis, statistical testing, feature analysis, pattern discovery, anomaly screening, and downloadable reports into one workflow.

The analytical core does **not** require an LLM, paid API, or database. After dependencies are installed, core computations can run locally without network access.

> **Sample data:** `sample_data/retail_sample.csv` is synthetic demonstration data. Metrics, charts, summaries, model results, and report findings are calculated from the dataset loaded in the application—not hardcoded.

## ✨ Key capabilities

| Area | What it does |
|---|---|
| **Data ingestion** | Loads CSV, Excel (`.xlsx`, `.xls`), JSON, and Parquet files, with format-specific validation and helpful load notes. |
| **Data profiling** | Summarizes dataset dimensions, data types, memory use, missing values, duplicate rows, unique values, near-constant columns, and potential identifiers. |
| **Cleaning & transformation** | Supports explicit missing-value handling, duplicate removal, type conversion, whitespace/category normalization, outlier clipping, scaling, and column exclusion. Keeps the original data and records transformations in an audit log. |
| **Exploratory analysis** | Provides interactive distributions, missingness visualizations, correlation heatmaps, and mixed-type descriptive summaries. |
| **Statistical analysis** | Includes Pearson, Spearman, and Kendall correlation; Welch’s t-test; Mann–Whitney U; chi-square independence; Shapiro–Wilk normality testing; supported confidence intervals and p-value corrections. |
| **Feature analysis** | Examines target associations, optional univariate mutual information, Random Forest evaluation on a holdout set, and permutation importance. |
| **Pattern discovery** | Applies K-means or DBSCAN to selected numeric features, provides a 2D projection for exploration, and calculates silhouette scores when valid. |
| **Anomaly screening** | Uses Isolation Forest to flag potentially unusual observations for review—not to declare them confirmed errors. |
| **Evidence-based insights** | Uses transparent local rules to highlight patterns such as missingness, duplicates, skew, rank correlations, dominant categories, and IQR-based outliers. |
| **Report center** | Exports HTML, PDF, and multi-sheet Excel reports using the current processed data, with selected charts, findings, methodology notes, and transformation history where supported. |

## 🧰 Technology stack

- **Language:** Python 3.10+
- **Interface:** Streamlit
- **Data handling:** Pandas, NumPy
- **Statistics:** SciPy, Statsmodels
- **Machine learning:** scikit-learn
- **Visualizations:** Plotly
- **Reports:** ReportLab, Excel workbook tooling
- **Quality assurance:** pytest

## 🚀 Quick start

### Requirements

- Python **3.10 or newer**; Python 3.11+ is recommended.
- Install the packages listed in `requirements.txt`.
- For Parquet support, `pyarrow` is required. Legacy `.xls` files require `xlrd`.

### Windows — PowerShell

```powershell
# Move into the project directory
cd data-science-analytics

# Install dependencies
py -3.11 -m pip install -r requirements.txt

# Start the application
py -3.11 -m streamlit run app.py
```

If the Python launcher is unavailable, use `python -m pip install -r requirements.txt` and `python -m streamlit run app.py` with the intended interpreter.

### Linux / macOS

```bash
cd data-science-analytics
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Streamlit normally displays the local address `http://localhost:8501` after startup.

## 🔄 Typical workflow

1. Open **Data Upload and Preview** and load a dataset or use the bundled synthetic sample.
2. Inspect the schema, data types, loader notes, missing values, and duplicates.
3. Open **Data Quality and Cleaning** and apply only the transformations appropriate to the analysis.
4. Review the transformation log. Use **Restore original dataset** to reset cleaning changes for the current session.
5. Explore distributions and descriptive summaries in **Exploratory Data Analysis**.
6. Select a suitable statistical test in **Statistical Analysis**, taking variable types and assumptions into account.
7. Choose a target in **Feature Analysis** to evaluate a model and inspect model-dependent permutation importance.
8. Use **Pattern Discovery** or **Anomaly Detection** where the data and sample size are appropriate.
9. Export results from **Report Center**.

## ⚙️ Configuration

The application supports environment-based defaults:

| Variable | Default / purpose |
|---|---|
| `DSA_MAX_UPLOAD_MB` | Application upload limit in MiB; defaults to 200 MB. This is not a guarantee that a file will fit in available RAM. |
| `DSA_PREVIEW_ROWS` | Preview-row default reserved for future UI refinements. |
| `DSA_CHART_SAMPLE_ROWS` | Default chart sample size, documented as `20000`. |
| `DSA_ALPHA` | Statistical significance level, defaults to `0.05`. |
| `DSA_RANDOM_STATE` | Reproducibility seed, defaults to `42`. |

To raise the accepted upload size, adjust Streamlit's own server upload limit as well as the application limit—for example through `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` or `.streamlit/config.toml`.

## 🧪 Testing and verification

Run the automated tests:

```bash
python -m pytest -q
```

Run the compile check:

```bash
python -m compileall -q app.py config core ui utils tests
```

The recorded verification reports **24 passing tests** across supported ingestion paths available in the build environment, cleaning, statistical calculations, target-based feature analysis, clustering/anomaly paths, and HTML/PDF/Excel report generation. A sample analysis was also generated; the PDF was visually inspected and the Excel workbook was reopened to check worksheets and chart objects.

See [`BUILD_STATUS.md`](BUILD_STATUS.md) for the full verification record and environment limitations.

## 🔬 Design principles & interpretation

- **Transparent operations:** cleaning actions are explicit, preserve the original DataFrame in session state, and append to a transformation log.
- **No invented metrics:** results are calculated from the active dataset.
- **Correlation is not causation:** associations may be affected by confounders, outliers, time, and missingness.
- **Statistical tests need context:** select tests based on measurement scale, study design, independence, and distributional assumptions. A p-value does not measure practical importance.
- **Feature importance is model-dependent:** it is not evidence of causation and depends on the split, model, and data.
- **Clusters are exploratory:** numeric clusters describe structure in selected feature space; they are not guaranteed real-world categories.
- **Anomaly flags are candidates for review:** they are not confirmed data errors.

## ⚠️ Limitations & data safety

- The Streamlit application was **not launched end-to-end in the recorded build environment**, because Streamlit was not installed there and package installation was blocked by network/DNS issues. Install `requirements.txt` in an environment with package access and run the commands above.
- Parquet ingestion could not be exercised in that environment because `pyarrow` was unavailable, although it is declared as a dependency. CSV, JSON, and `.xlsx` ingestion have test coverage. Legacy `.xls` additionally relies on `xlrd`.
- The application generally holds the dataset in memory; this is not a distributed-processing system. Large or very wide datasets can require substantial memory and compute.
- The upload cap is an application guard, not a promise that any file of that size can be processed on every machine.
- Uploaded content is parsed as data and is not intentionally executed as Python or shell commands. However, session memory and downloaded reports can contain sensitive records. Only upload data you are authorized to process.

## 📁 Project structure

```text
data-science-analytics/
├── app.py
├── requirements.txt
├── BUILD_STATUS.md
├── config/
│   └── settings.py
├── core/
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── data_cleaner.py
│   ├── profiler.py
│   ├── statistical_analysis.py
│   ├── feature_analysis.py
│   ├── pattern_discovery.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   ├── insights_engine.py
│   └── report_generator.py
├── ui/
├── utils/
├── tests/
└── sample_data/
```

## 🤝 Contributing

Issues, bug reports, and focused improvements are welcome. When reporting a problem, include the Python version, relevant dependency versions, reproducible steps, and a small non-sensitive sample when possible. Please do not commit private datasets, generated reports containing sensitive data, credentials, or API keys.

---

<a id="فارسی"></a>
# 🇮🇷 فارسی

<div dir="rtl" align="right">

## معرفی پروژه

**پلتفرم تحلیل علم داده** یک محیط تحلیلی ماژولار و محلی است که با رابط تعاملی Streamlit، فرایند بارگذاری، اعتبارسنجی، پاک‌سازی، تحلیل و گزارش‌گیری از مجموعه‌داده‌ها را در یک گردش‌کار یکپارچه قرار می‌دهد. این پروژه امکاناتی برای بررسی کیفیت داده، تحلیل اکتشافی، آزمون‌های آماری، تحلیل ویژگی‌ها، کشف الگو، شناسایی داده‌های غیرعادی و تولید گزارش‌های قابل دانلود فراهم می‌کند.

هستهٔ محاسباتی پروژه به **مدل زبانی، API پولی یا پایگاه داده نیاز ندارد**. پس از نصب وابستگی‌ها، تحلیل‌های اصلی می‌توانند به‌صورت محلی و بدون اتصال شبکه اجرا شوند.

> **دربارهٔ دادهٔ نمونه:** فایل `sample_data/retail_sample.csv` داده‌ای مصنوعی و صرفاً نمایشی است. آمارها، نمودارها، خلاصه‌ها، نتایج مدل و گزارش‌ها بر اساس داده‌ای محاسبه می‌شوند که در برنامه بارگذاری شده است؛ نتایج از پیش ثابت‌گذاری نشده‌اند.

## ✨ قابلیت‌های اصلی

| بخش | توضیحات |
|---|---|
| **بارگذاری داده** | پشتیبانی از CSV، فایل‌های Excel با پسوندهای `.xlsx` و `.xls`، فرمت JSON و Parquet همراه با اعتبارسنجی و پیام‌های راهنما. |
| **پروفایل‌سازی داده** | بررسی ابعاد داده، نوع ستون‌ها، تخمین حافظه، مقادیر گمشده، ردیف‌های تکراری، مقادیر یکتا، ستون‌های تقریباً ثابت و شناسه‌های احتمالی. |
| **پاک‌سازی و تبدیل** | مدیریت صریح مقادیر گمشده، حذف تکراری‌ها، تبدیل نوع داده، یکسان‌سازی فاصله‌ها و دسته‌ها، محدودسازی داده‌های پرت، مقیاس‌بندی و حذف ستون. دادهٔ اصلی حفظ می‌شود و تغییرات در گزارش ثبت می‌شوند. |
| **تحلیل اکتشافی** | نمایش توزیع‌ها، نمودار مقادیر گمشده، نقشهٔ حرارتی هم‌بستگی و خلاصه‌های توصیفی برای داده‌های ترکیبی. |
| **تحلیل آماری** | هم‌بستگی Pearson، Spearman و Kendall؛ آزمون Welch؛ آزمون Mann–Whitney U؛ آزمون استقلال کای‌دو؛ آزمون نرمال‌بودن Shapiro–Wilk؛ و قابلیت‌های پیاده‌سازی‌شده برای فاصلهٔ اطمینان و اصلاح مقدار p. |
| **تحلیل ویژگی‌ها** | بررسی ارتباط ویژگی‌ها با متغیر هدف، امتیاز اطلاعات متقابل تک‌متغیره به‌صورت اختیاری، ارزیابی Random Forest روی دادهٔ نگه‌داشته‌شده و اهمیت جایگشتی ویژگی‌ها. |
| **کشف الگو** | اجرای K-means یا DBSCAN روی ویژگی‌های عددی انتخاب‌شده، نمایش دوبعدی برای بررسی الگوها و محاسبهٔ امتیاز سیلوئت در شرایط معتبر. |
| **بررسی ناهنجاری‌ها** | استفاده از Isolation Forest برای مشخص‌کردن موارد غیرعادی احتمالی جهت بررسی بیشتر؛ این موارد الزاماً خطای قطعی نیستند. |
| **بینش‌های مبتنی بر شواهد** | استفاده از قواعد محلی و قابل‌فهم برای برجسته‌کردن مقادیر گمشده، تکراری‌ها، چولگی، هم‌بستگی رتبه‌ای، دسته‌های غالب و داده‌های پرت بر پایهٔ IQR. |
| **مرکز گزارش** | خروجی HTML، PDF و Excel چندبرگه‌ای بر اساس دادهٔ پردازش‌شده، همراه با نمودارها و توضیحات روش‌شناسی و تاریخچهٔ تبدیل‌ها در بخش‌های پشتیبانی‌شده. |

## 🧰 فناوری‌های استفاده‌شده

- **زبان برنامه‌نویسی:** Python 3.10 به بالا
- **رابط کاربری:** Streamlit
- **پردازش داده:** Pandas و NumPy
- **آمار:** SciPy و Statsmodels
- **یادگیری ماشین:** scikit-learn
- **نمودارها:** Plotly
- **گزارش‌گیری:** ReportLab و ابزارهای ساخت فایل Excel
- **تست:** pytest

## 🚀 نصب و اجرا

### پیش‌نیازها

- Python نسخهٔ **3.10 یا جدیدتر**؛ نسخهٔ 3.11 به بالا پیشنهاد می‌شود.
- نصب وابستگی‌های موجود در `requirements.txt`.
- برای خواندن Parquet به `pyarrow` و برای فایل‌های قدیمی `.xls` به `xlrd` نیاز است.

### ویندوز — PowerShell

```powershell
# ورود به پوشهٔ پروژه
cd data-science-analytics

# نصب وابستگی‌ها
py -3.11 -m pip install -r requirements.txt

# اجرای برنامه
py -3.11 -m streamlit run app.py
```

اگر دستور `py` در دسترس نیست، از `python -m pip install -r requirements.txt` و سپس `python -m streamlit run app.py` با مفسر پایتون موردنظر استفاده کنید.

### لینوکس و macOS

```bash
cd data-science-analytics
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

پس از اجرا، آدرس محلی برنامه معمولاً `http://localhost:8501` خواهد بود.

## 🔄 گردش‌کار پیشنهادی

۱. در بخش **Data Upload and Preview** یک فایل بارگذاری کنید یا از دادهٔ نمونهٔ پروژه استفاده کنید.

۲. ساختار ستون‌ها، نوع داده‌ها، پیام‌های بارگذاری، مقادیر گمشده و ردیف‌های تکراری را بررسی کنید.

۳. در بخش **Data Quality and Cleaning** فقط تبدیل‌های مناسب برای مسئلهٔ تحلیلی خود را اعمال کنید.

۴. گزارش تغییرات را بررسی کنید. با گزینهٔ **Restore original dataset** می‌توانید تغییرات پاک‌سازی همان نشست را به حالت اولیه برگردانید.

۵. در **Exploratory Data Analysis** توزیع‌ها و آمارهای توصیفی را بررسی کنید.

۶. در **Statistical Analysis** آزمون آماری متناسب با نوع متغیرها و فرض‌های روش را انتخاب کنید.

۷. در **Feature Analysis** متغیر هدف را انتخاب کرده و ارزیابی مدل و اهمیت ویژگی‌ها را بررسی کنید.

۸. در صورت مناسب‌بودن حجم و ساختار داده، از **Pattern Discovery** یا **Anomaly Detection** استفاده کنید.

۹. گزارش‌های نهایی را از **Report Center** دریافت کنید.

## ⚙️ تنظیمات

مقادیر پیش‌فرض از طریق متغیرهای محیطی قابل تنظیم‌اند:

| متغیر | کاربرد یا مقدار پیش‌فرض |
|---|---|
| `DSA_MAX_UPLOAD_MB` | سقف بارگذاری برنامه بر حسب MiB؛ مقدار پیش‌فرض ۲۰۰ مگابایت است. این مقدار تضمین نمی‌کند که هر فایل با این اندازه در حافظهٔ سیستم جا شود. |
| `DSA_PREVIEW_ROWS` | مقدار پیش‌فرض مربوط به پیش‌نمایش که برای اصلاحات آیندهٔ رابط رزرو شده است. |
| `DSA_CHART_SAMPLE_ROWS` | اندازهٔ پیش‌فرض نمونه برای نمودارها؛ مقدار مستندشده `20000` است. |
| `DSA_ALPHA` | سطح معناداری آماری؛ مقدار پیش‌فرض `0.05` است. |
| `DSA_RANDOM_STATE` | بذر تکرارپذیری؛ مقدار پیش‌فرض `42` است. |

برای افزایش حجم مجاز فایل، محدودیت بارگذاری خود Streamlit را نیز همراه با محدودیت برنامه تنظیم کنید؛ برای مثال از `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` یا فایل `.streamlit/config.toml` استفاده کنید.

## 🧪 تست و وضعیت اعتبارسنجی

اجرای تست‌های خودکار:

```bash
python -m pytest -q
```

بررسی کامپایل فایل‌های پروژه:

```bash
python -m compileall -q app.py config core ui utils tests
```

طبق گزارش ثبت‌شده، **۲۴ تست با موفقیت اجرا شده‌اند**. این تست‌ها مسیرهای بارگذاری پشتیبانی‌شده در محیط ساخت، پاک‌سازی، محاسبات آماری، تحلیل ویژگی بر اساس متغیر هدف، مسیرهای خوشه‌بندی و تشخیص ناهنجاری و تولید گزارش HTML/PDF/Excel را پوشش می‌دهند. یک تحلیل نمونه نیز تولید شده؛ فایل PDF از نظر بصری بررسی شده و فایل Excel برای کنترل برگه‌ها و نمودارها دوباره باز شده است.

برای جزئیات کامل و محدودیت‌های محیط ساخت، فایل [`BUILD_STATUS.md`](BUILD_STATUS.md) را ببینید.

## 🔬 اصول طراحی و تفسیر نتایج

- **عملیات شفاف:** مراحل پاک‌سازی به‌صورت صریح اجرا می‌شوند؛ دادهٔ اصلی در وضعیت نشست حفظ می‌شود و تغییرات در گزارش ثبت می‌شوند.
- **بدون آمار ساختگی:** نتایج از دادهٔ فعال بارگذاری‌شده محاسبه می‌شوند.
- **هم‌بستگی به معنی علیت نیست:** عوامل مخدوش‌کننده، داده‌های پرت، زمان و مقادیر گمشده ممکن است بر ارتباط مشاهده‌شده اثر بگذارند.
- **آزمون آماری به زمینه نیاز دارد:** انتخاب آزمون باید با مقیاس اندازه‌گیری، طراحی مطالعه، استقلال مشاهدات و فرض‌های توزیعی هماهنگ باشد. مقدار p به‌تنهایی اهمیت عملی را نشان نمی‌دهد.
- **اهمیت ویژگی وابسته به مدل است:** اهمیت ویژگی اثبات رابطهٔ علت‌ومعلولی نیست و به داده، مدل و تقسیم‌بندی وابسته است.
- **خوشه‌ها اکتشافی‌اند:** خوشه‌بندی ویژگی‌های عددی، ساختار موجود در فضای ویژگی انتخاب‌شده را نشان می‌دهد و لزوماً معادل دسته‌های واقعی نیست.
- **پرچم ناهنجاری نیازمند بررسی است:** دادهٔ علامت‌گذاری‌شده الزاماً خطا محسوب نمی‌شود.

## ⚠️ محدودیت‌ها و امنیت داده

- در محیط ساخت ثبت‌شده، برنامهٔ Streamlit به‌صورت کامل از ابتدا تا انتها اجرا نشد؛ زیرا Streamlit نصب نبود و نصب وابستگی‌های جدید به دلیل مشکل شبکه/DNS ممکن نشد. وابستگی‌ها را در محیطی با دسترسی به بسته‌ها نصب کرده و برنامه را طبق دستورهای بالا اجرا کنید.
- به دلیل نبود `pyarrow` در آن محیط، خواندن Parquet عملاً آزمایش نشد؛ هرچند این وابستگی در `requirements.txt` تعریف شده است. بارگذاری CSV، JSON و `.xlsx` تست شده و فایل `.xls` قدیمی به `xlrd` نیاز دارد.
- داده‌ها عموماً در حافظه نگهداری می‌شوند؛ بنابراین پروژه سامانهٔ پردازش توزیع‌شده نیست و مجموعه‌داده‌های بزرگ یا بسیار عریض ممکن است به حافظه و توان پردازشی زیادی نیاز داشته باشند.
- سقف بارگذاری صرفاً محدودیتی در سطح برنامه است و تضمین نمی‌کند هر فایل با آن اندازه در همهٔ سیستم‌ها پردازش شود.
- محتوای بارگذاری‌شده به‌عنوان داده پردازش می‌شود و قرار نیست به‌عنوان دستور Python یا shell اجرا شود؛ بااین‌حال حافظهٔ نشست و گزارش‌های دانلودشده ممکن است شامل اطلاعات حساس باشند. فقط داده‌هایی را بارگذاری کنید که مجاز به پردازش آن‌ها هستید.

## 📁 ساختار پروژه

```text
data-science-analytics/
├── app.py
├── requirements.txt
├── BUILD_STATUS.md
├── config/
│   └── settings.py
├── core/
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── data_cleaner.py
│   ├── profiler.py
│   ├── statistical_analysis.py
│   ├── feature_analysis.py
│   ├── pattern_discovery.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   ├── insights_engine.py
│   └── report_generator.py
├── ui/
├── utils/
├── tests/
└── sample_data/
```

## 🤝 مشارکت در توسعه

گزارش باگ، پیشنهاد بهبود و مشارکت‌های مشخص و قابل بررسی ارزشمند هستند. هنگام ثبت مشکل، نسخهٔ Python، نسخه‌های وابستگی مرتبط، مراحل بازتولید و در صورت امکان نمونه‌ای کوچک و فاقد اطلاعات حساس را ارائه کنید. داده‌های خصوصی، گزارش‌های حاوی اطلاعات حساس، اطلاعات ورود و کلیدهای API را به مخزن اضافه نکنید.

</div>

---

<div align="center">

**Built for careful analysis — ساخته‌شده برای تحلیل دقیق**

[GitHub](https://github.com/PyQubit) · [Portfolio](https://pyqubit.github.io/)

</div>
