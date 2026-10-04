# Lab 09 Hands-on — MLflow + GitHub Actions (Breast Cancer)

> ใบงาน: ดัดแปลง pipeline จากเอกสารประกอบ (ชุด Wine) มาใช้กับชุด **Breast Cancer**
> โฟลเดอร์ในเครื่อง: `D:\ยำdowload\mlops_cancer_submit` → GitHub: `Omayata/Lab9_mlops_cancer_1`

---

## 0. ภาพรวม

```
explore.py          01_data_validation.py      02_data_preprocessing.py     03_train_evaluate_register.py      04_load_and_predict.py
(สำรวจข้อมูล)  ──▶  ตรวจข้อมูล               ──▶ แบ่ง 75:25 stratify     ──▶ Scaler+LogReg                ──▶ โหลด @staging
569×31, 2 คลาส      2 คลาส? balance ≥ 20%?       log csv เป็น artifact         acc ≥ 0.95 และ AUC ≥ 0.98       ทำนาย malignant 1 ราย
                    ไม่ผ่าน → exit code 1         พิมพ์ Run ID ─────────────▶   → register + alias @staging      + benign 1 ราย
```

| | Wine (เอกสารประกอบ) | Breast Cancer (ใบงานนี้) |
|---|---|---|
| คำสั่งโหลด | `load_wine` | `load_breast_cancer` |
| ขนาด | 178 × 14 | **569 × 31** (30 ฟีเจอร์ + target) |
| คลาส | 3 | **2** (0 = malignant, 1 = benign) |
| คลาสน้อยสุด | 26.97% | **37.26%** (malignant) |
| ตัวชี้วัด | accuracy | **accuracy + ROC-AUC** |
| ชื่อโมเดล | wine-classifier-prod | **cancer-classifier-prod** |

---

## 1. ไฟล์แต่ละไฟล์ทำอะไร

```
mlops_cancer_submit/
├── scripts/
│   ├── explore.py                     ส่วน 1: สำรวจข้อมูล (Capture 1.2)
│   ├── 01_data_validation.py          ส่วน 2.1: ตรวจข้อมูล (Capture 2.1, 2.2)
│   ├── 02_data_preprocessing.py       ส่วน 2.2: แบ่งข้อมูล + log artifact (Capture 2.3)
│   ├── 03_train_evaluate_register.py  ส่วน 3.1: เทรน + gate + register (Capture 3.1)
│   └── 04_load_and_predict.py         ส่วน 3.2: โหลดโมเดลจาก Registry มาทำนาย (Capture 3.2)
├── tests/
│   ├── test_data.py                   pytest ตรวจข้อมูล (ใช้ใน CI)
│   └── test_model.py                  pytest model gate (ใช้ใน CI)
├── .github/workflows/
│   ├── hello-actions.yml              ส่วน 4.1: workflow แรก (Capture 4.1)
│   └── ci.yml                         ส่วน 4.2: CI 3 ด้าน (Capture 4.2)
├── requirements.txt                   library ที่ต้องติดตั้ง (มี ruff ด้วย)
├── pyproject.toml                     ตั้งค่า ruff
├── .gitignore                         ไม่ commit mlflow.db / mlruns/ / processed_data/
│
│   ── สร้างขึ้นเองตอนรัน (ไม่ขึ้น GitHub) ──
├── mlflow.db                          ฐานข้อมูล MLflow (run, param, metric, registry)
├── mlruns/                            ไฟล์ artifact (csv, โมเดล)
├── processed_data/                    train.csv / test.csv ที่ขั้น 02 เขียน
└── _backup_old_runs/                  ผลรันเก่าก่อนแก้ (ลบทิ้งได้ถ้าไม่ต้องการ)
```

### `scripts/explore.py`
พิมพ์ shape, ชื่อคลาส, จำนวนคลาส, สัดส่วนคลาส → ต้องได้ `(569, 31)`, `['malignant' 'benign']`, `1 → 0.6274`, `0 → 0.3726`

### `scripts/01_data_validation.py`
- experiment: `Breast Cancer - Data Validation`
- เช็ค 3 อย่าง: ไม่มี missing, **จำนวนคลาสต้องเท่ากับ 2**, **คลาสน้อยสุด ≥ 20%** (`MIN_CLASS_BALANCE = 0.20`)
- log metric: `num_rows`, `num_cols`, `missing_values`, **`class_balance`** · param: `num_classes`, `validation_status`
- ไม่ผ่าน → `raise SystemExit(...)` = **exit code 1** → CI แดงและหยุด
- `class_balance` คำนวณจาก `value_counts(normalize=True).min()` = สัดส่วนของคลาสที่น้อยที่สุดจริง ๆ

### `scripts/02_data_preprocessing.py`
- experiment: `Breast Cancer - Data Preprocessing`
- `train_test_split(test_size=0.25, stratify=y, random_state=42)` → train **426** / test **143**
- เซฟ `processed_data/train.csv`, `test.csv` แล้ว `mlflow.log_artifacts(...)` ขึ้น MLflow
- **พิมพ์ `Preprocessing Run ID`** (hex 32 ตัว) → copy ไปใช้ขั้น 03
- ไม่ scale ในขั้นนี้ — scaler ไปอยู่ใน Pipeline ขั้น 03

### `scripts/03_train_evaluate_register.py`
รัน: `python scripts/03_train_evaluate_register.py <RUN_ID> <C>`
1. `download_artifacts(run_id=RUN_ID, artifact_path="processed_data")` → อ่าน csv **จาก run ของขั้น 02**
2. `Pipeline([StandardScaler, LogisticRegression(C)])`
3. คำนวณ `accuracy` และ `roc_auc` (จาก `predict_proba(X_test)[:, 1]`) → log ทั้งคู่
4. `log_model(name="cancer_classifier_pipeline", input_example=...)`
5. **Gate: ต้องผ่านทั้ง `accuracy ≥ 0.95` และ `roc_auc ≥ 0.98`** → `register_model` ชื่อ `cancer-classifier-prod` + ตั้ง alias `@staging`
   ถ้าไม่ผ่านข้อใดข้อหนึ่ง → log ไว้แต่ **ไม่ลงทะเบียน**

ผลที่ทดลองแล้ว (split เดียวกัน):

| C | Accuracy | ROC-AUC | ผ่าน gate? |
|---|---|---|---|
| 0.001 | 0.9161 | 0.9904 | ❌ (acc ไม่ผ่าน) |
| 0.01 | 0.9371 | 0.9952 | ❌ (acc ไม่ผ่าน แม้ AUC ผ่าน) |
| 0.1 | 0.9790 | 0.9971 | ✅ |
| 1.0 | 0.9860 | 0.9977 | ✅ |
| **10.0 (ใบงาน)** | **0.9720** | **0.9945** | ✅ |

### `scripts/04_load_and_predict.py`
- `mlflow.pyfunc.load_model("models:/cancer-classifier-prod@staging")`
- หยิบรายแรกของแต่ละคลาส: แถว 0 (malignant) และแถว 19 (benign)
- พิมพ์ Actual / Predicted **เป็นคำ** และ Correct: True/False
- ส่งข้อมูลดิบได้เลย เพราะ scaler อยู่ใน pipeline

### `tests/test_data.py` และ `tests/test_model.py`
- data: shape (569, 31), ไม่มี missing, 2 คลาส, balance ≥ 20%, `mean radius` อยู่ในช่วง 5–30
- model: เทรน C=10 → accuracy ≥ 0.95, ROC-AUC ≥ 0.98, มี `scaler` ใน pipeline, predict 5 แถวได้ shape (5,)

### `.github/workflows/hello-actions.yml`
workflow ตัวอย่าง: trigger `push` + `workflow_dispatch` → echo `github.event_name`, `runner.os`, `github.repository` แล้ว checkout + `ls`

### `.github/workflows/ci.yml`
| ข้อกำหนดในใบงาน | อยู่ตรงไหน |
|---|---|
| trigger push main / pull_request / workflow_dispatch | `on:` |
| job `lint` รัน `ruff check scripts/ tests/` บน 3.12 | `jobs.lint` |
| job `test` มี `needs: lint` + matrix 3.11, 3.12 | `jobs.test` |
| step แยก Data validation → Data tests → Model quality gate | steps ใน `test` |
| upload-artifact@v4 + `if: always()` | step `Upload test reports` |

---

## 2. ติดตั้ง (ครั้งเดียว)

**Windows (Anaconda Prompt):**
```bat
conda create -n mlflow_env python=3.12 -y
conda activate mlflow_env
cd /d "D:\ยำdowload\mlops_cancer_submit"
pip install -r requirements.txt
```
**macOS:** เหมือนกัน แต่ถ้า `conda activate` ไม่ได้ให้รัน `conda init zsh` แล้วเปิด terminal ใหม่

> ลงทีละตัวต้องใส่ฟันหนู: `pip install "ruff>=0.8"` — ไม่งั้น `>` จะกลายเป็นคำสั่งเขียนลงไฟล์ (ได้ไฟล์ขยะชื่อ `0.8`)

**Capture 1.1** — ตรวจเวอร์ชันและ tracking URI:
```bat
python -c "import mlflow, sklearn; print(mlflow.__version__, sklearn.__version__); print(mlflow.get_tracking_uri())"
```
ควรเห็นเช่น `3.16.0 1.9.0` และ `sqlite:///mlflow.db`

---

## 3. ลำดับการรัน + สิ่งที่ต้องเห็น (ตรงกับ Capture)

> **ทุกคำสั่งรันจากโฟลเดอร์ `mlops_cancer_submit`** — `mlflow.db` อยู่ที่นี่ ถ้ารันผิดโฟลเดอร์จะได้ db ใหม่และหาโมเดลไม่เจอ

| ลำดับ | คำสั่ง | ต้องเห็น | Capture |
|---|---|---|---|
| 1 | `python scripts/explore.py` | `(569, 31)` · `['malignant' 'benign']` · `0.6274 / 0.3726` | 1.2 |
| 2 | `python scripts/01_data_validation.py` | `569 rows, 31 columns` · `Number of classes: 2` · `Validation status: Success` | 2.1 |
| 3 | (ทดสอบให้ fail) ดูข้อ 3.1 ด้านล่าง | `FAILED: class balance 37.26% is below 45%` · exit code = 1 | 2.2 |
| 4 | `python scripts/02_data_preprocessing.py` | `training_set_rows = 426` · `Preprocessing Run ID: <hex 32 ตัว>` | 2.3 |
| 5 | `python scripts/03_train_evaluate_register.py <RUN_ID> 10.0` | `Accuracy: 0.9720` · `ROC-AUC: 0.9945` · `version 1` · `@staging` | 3.1 |
| 6 | `python scripts/04_load_and_predict.py` | 2 แถว: malignant→malignant, benign→benign, `Correct: True` ทั้งคู่ | 3.2 |
| 7 | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Models → cancer-classifier-prod → Aliases มี `@staging` | 3.3 |

### 3.1 Capture 2.2 — ทำให้ validation ล้มเหลวชั่วคราว
1. เปิด `scripts/01_data_validation.py` แก้ `MIN_CLASS_BALANCE = 0.20` → `0.45`
2. รัน แล้ว**เช็ค exit code ทันที**:
   ```bat
   python scripts/01_data_validation.py
   echo %ERRORLEVEL%
   ```
   - PowerShell ใช้ `echo $LASTEXITCODE` · macOS ใช้ `echo $?`
   - ต้องได้ **1** (ไม่ใช่ 0)
3. **แก้กลับเป็น `0.20`** แล้วรันอีกครั้งให้ได้ Success

### 3.2 ถ้า version ไม่ใช่ 1
ทุกครั้งที่ register ผ่าน version จะเพิ่ม (2, 3, …) ถ้าอยากได้ version 1 ตามใบงาน ให้เริ่มใหม่จาก db เปล่า:
ปิด MLflow UI ก่อน แล้วย้าย/ลบ `mlflow.db` กับ `mlruns/` แล้วรันขั้น 01 → 04 ใหม่

---

## 4. ดูผลใน MLflow UI

```bat
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
เปิด http://127.0.0.1:5000 (macOS เติม `--port 5001` แล้วเปิด :5001 เพราะ AirPlay ยึด 5000) · ปิดด้วย `Ctrl+C`

| ดูตรงไหน | เห็นอะไร |
|---|---|
| Experiments → `Breast Cancer - Data Validation` | metric `class_balance` = 0.3726, `num_rows`, param `validation_status` |
| Experiments → `Breast Cancer - Data Preprocessing` → run → **Artifacts** | โฟลเดอร์ `processed_data/` (train.csv, test.csv) |
| Experiments → `Breast Cancer - Model Training` | run `logistic_regression_C_10.0` → metrics `accuracy`, `roc_auc` / Artifacts → โมเดล |
| เลือกหลาย run → **Compare** | เทียบ C กับ accuracy / roc_auc |
| แท็บ **Models** → `cancer-classifier-prod` | ทุก version + คอลัมน์ **Aliases** (`@staging`) ← Capture 3.3 |

---

## 5. Deploy เป็น REST API (เพิ่มเติม)

**Terminal 1:**
```bat
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/cancer-classifier-prod@staging" -p 5002 --env-manager local
```
(PowerShell: `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"` · macOS: `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db`)

**Terminal 2** — ส่งข้อมูลแถวแรก (malignant):
```python
import requests
from sklearn.datasets import load_breast_cancer
X = load_breast_cancer(as_frame=True).data
row = X.iloc[[0]]
payload = {"dataframe_split": {"columns": list(row.columns), "data": row.values.tolist()}}
r = requests.post("http://127.0.0.1:5002/invocations", json=payload)
print(r.json())   # {'predictions': [0]}  → 0 = malignant
```

---

## 6. ส่งขึ้น GitHub + ดู Actions (ส่วนที่ 4)

repo นี้ผูกกับ GitHub ไว้แล้ว แค่ commit แล้ว push:
```bat
git add -A
git commit -m "lab09 hands-on: fix to match worksheet (breast cancer)"
git push
```
(ถ้าเริ่มโปรเจกต์ใหม่จากศูนย์: `git init` → `git add .` → `git commit -m "..."` → `git branch -M main` → `git remote add origin https://github.com/<user>/<repo>.git` → `git push -u origin main` — ต้องมี `.gitignore` ก่อน `git add .`)

ตรวจก่อน push ว่า CI จะผ่าน:
```bat
ruff check scripts/ tests/
pytest tests/ -v
```

ดูผลบน GitHub → แท็บ **Actions**:
- **Capture 4.1**: workflow `GitHub Actions Demo` เขียว → คลิก job → ใน log ต้องเห็นชื่อ repo จริงและคำว่า `Linux` (ไม่ใช่ `${{ ... }}`)
- **Capture 4.2**: workflow `CI` → ต้องเห็น `lint`, `test (3.11)`, `test (3.12)` เขียวครบ และด้านล่างหน้าสรุปมีส่วน **Artifacts** (`reports-py3.11`, `reports-py3.12`)
- รัน CI เองได้: Actions → CI → **Run workflow** (เพราะมี `workflow_dispatch`)

---

## 7. คำตอบคำถามในใบงาน (สรุป)

**1.1 จุดในโค้ด Wine ที่จะผิดถ้าใช้กับชุดนี้โดยไม่แก้**
- `if missing_values > 0 or num_classes < 3: Failed` → ชุดนี้มี 2 คลาส จะ **Failed ทันทีทั้งที่ข้อมูลปกติ** ต้องเปลี่ยนเป็น `num_classes != 2`
- test เช็ค 14 คอลัมน์ / 3 คลาส / ช่วง `alcohol` → ผิดหมด (ชุดนี้ 31 คอลัมน์, 2 คลาส, ไม่มีคอลัมน์ alcohol)
- ใช้ accuracy อย่างเดียว ไม่พอสำหรับ binary ที่คลาสไม่สมดุล → ต้องเพิ่ม ROC-AUC
- ชื่อ experiment / ชื่อโมเดล `wine-...` ต้องเปลี่ยน

**2.1 ทำไม log train/test เป็น artifact แทนให้ขั้นถัดไปโหลดเอง**
ขั้นเทรนจะใช้ข้อมูลชุดเดียวกับที่ขั้นเตรียมข้อมูลแบ่งไว้จริง (อ้างด้วย Run ID) → ย้อนรอยได้ว่าโมเดลเทรนจากข้อมูลชุดไหน (lineage), ทำซ้ำได้ (reproducible), และถ้าขั้นเตรียมข้อมูลเปลี่ยน (เช่น เปลี่ยน split/ทำความสะอาด) ก็ไม่ต้องไปแก้โค้ดซ้ำในขั้นเทรน

**2.2 `stratify=y` มีไว้ทำอะไร**
ทำให้สัดส่วนคลาสใน train และ test เท่ากับข้อมูลทั้งหมด (≈ 37% malignant) ถ้าไม่ใส่ การสุ่มอาจได้ test ที่ malignant มาก/น้อยผิดปกติ ผล accuracy/AUC จะแกว่งและไม่สะท้อนของจริง — สำคัญกับชุดนี้เพราะคลาสไม่สมดุล (63:37) และ test มีแค่ 143 แถว

**3.1 ทำไมต้องดู ROC-AUC ด้วย และทำไม Wine ไม่ทำ**
accuracy ดูที่ threshold 0.5 จุดเดียว และถูกคลาสใหญ่ครอบงำได้ (ทายว่า benign หมดก็ได้ ~63% แล้ว) ส่วน ROC-AUC วัดความสามารถในการ "จัดอันดับ" ผู้ป่วยกับไม่ป่วยที่ทุก threshold ซึ่งสำคัญในงานแพทย์ที่อาจปรับ threshold เพื่อลด false negative
Wine เป็น 3 คลาส — `roc_auc_score` แบบ binary ใช้ตรง ๆ ไม่ได้ (ต้องทำ one-vs-rest เพิ่ม) และคลาสค่อนข้างสมดุล accuracy จึงพอ

**3.2 ถ้า roc_auc = 0.97 (ไม่ผ่าน) แต่ accuracy = 0.96 (ผ่าน)**
เงื่อนไขเป็น `acc_pass and auc_pass` → **ไม่ลงทะเบียน** (พิมพ์ `QUALITY GATE: FAILED`) แต่ run และ metric ยังถูก log ไว้ดูย้อนหลัง
ออกแบบให้ต้องผ่านทุกเกณฑ์เพราะแต่ละตัววัดคนละมุม — ผ่านแค่ตัวเดียวแปลว่ายังมีจุดอ่อน ควรกันไว้ไม่ให้โมเดลที่ไม่มั่นใจขึ้น production (ลองดูได้จริงด้วย `C=0.01`: acc 0.937 ไม่ผ่าน แต่ AUC 0.995 ผ่าน → ไม่ register)

---

## 8. ปัญหาที่เจอบ่อย

| อาการ | วิธีแก้ |
|---|---|
| ขั้น 03 `Error loading artifacts` | Run ID ผิด/ไม่ครบ หรือใช้ Run ID ของขั้น 01 แทน 02 หรือรันคนละโฟลเดอร์ |
| ขั้น 04 `Error loading model` | ยังไม่มี version ไหนผ่าน gate (เช่นรัน C=0.01 อย่างเดียว) → รัน 03 ด้วย C=10 |
| UI ไม่เห็น run | ลืม `--backend-store-uri sqlite:///mlflow.db` หรือเปิดจากโฟลเดอร์อื่น |
| CI แดงที่ ruff | ลำดับ import / ตัวแปรไม่ได้ใช้ → รัน `ruff check scripts/ tests/ --fix` |
| CI แดงที่ pytest หาไฟล์ไม่เจอ | ชื่อไฟล์ test ต้องขึ้นต้นด้วย `test_` และอยู่ใน `tests/` |
| port ชน | UI 5000, serve 5002 หรือเปลี่ยน `-p` |

---

## 9. สิ่งที่แก้จากเวอร์ชันเดิมบน GitHub

| เดิม | แก้เป็น |
|---|---|
| 01 ไม่ log MLflow, เช็ค balance ด้วยตัวแปรค้างจากลูป (ดูแค่คลาส benign), เกณฑ์ 45% | log MLflow + metric `class_balance` จากคลาสน้อยสุดจริง, เกณฑ์ 20%, `SystemExit` |
| 02 scale + เซฟ `scaler.joblib` แยก, experiment ไม่ตรงกับ 03 | ไม่ scale, log `processed_data/` เป็น artifact, experiment แยกตามขั้น |
| 03 อ่าน csv ในเครื่อง ไม่ได้ใช้ Run ID, ไม่มี Pipeline | `download_artifacts` ด้วย Run ID, Pipeline (Scaler + LogReg), gate 2 เกณฑ์ |
| 04 ต้องโหลด scaler มา transform เอง | pyfunc + pipeline ส่งข้อมูลดิบได้เลย |
| `ci.yml` ว่าง, test เป็นของ Wine, ชื่อไฟล์ `test.data.py` | CI ครบตามใบงาน, test ของ Breast Cancer, ชื่อ `test_data.py` |
| ไม่มี `requirements.txt` / `pyproject.toml` (CI install ไม่ได้) | เพิ่มครบ |
| commit โฟลเดอร์ `artifacts/` ขึ้น GitHub | ลบออก, `.gitignore` กัน `processed_data/`, `mlflow.db`, `mlruns/` |
