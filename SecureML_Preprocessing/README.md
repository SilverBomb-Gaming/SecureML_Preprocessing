# SecureML Preprocessing 🚀

**A feature engineering and preprocessing pipeline for secure machine learning models.**  
- ✅ **Handles missing values** (mean, median, mode)  
- ✅ **Feature scaling** (standardization & normalization)  
- ✅ **Encodes categorical data** (one-hot & label encoding)  
- ✅ **Selects relevant features** (removes low-variance & correlated ones)  
- ✅ **Exports a clean CSV for ML models**  

## 🔧 Installation  
git clone https://github.com/SilverBomb-Gaming/SecureML_Preprocessing.git
cd SecureML_Preprocessing
pip install -r requirements.txt

To preprocess a dataset, simply run: 
Python main.py
This will generate a clean datatset and save it as preprocessed_data.csv

SAMPLE OUTPUT:

_Original Data:
    feature1  feature2 category
0        10       100        A
1        20       200        B
2        30       300        A
3        40       400        C
4        50       500        B

Scaled Data:
    feature1  feature2
0 -1.414214 -1.414214
1 -0.707107 -0.707107
2  0.000000  0.000000
3  0.707107  0.707107
4  1.414214  1.414214

Encoded Data:
    feature1  feature2  category
0        10       100         0
1        20       200         1
2        30       300         0
3        40       400         2
4        50       500         1

📂 **Preprocessed dataset saved as**: `preprocessed_data.csv`

📌 Features & Customization
Modify main.py to:
✅ Change feature scaling methods (standardization, min-max)
✅ Customize categorical encoding (one-hot, label encoding)
✅ Handle missing values differently

📢 Connect With Me
🔗 GitHub: SilverBomb-Gaming
🔗 LinkedIn: SilverBomb Gaming

⭐ Show Some Love!
If you found this helpful, star the repo ⭐ on GitHub! 💙
