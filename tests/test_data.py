"""Data tests — ตรวจว่าข้อมูลยังหน้าตาเหมือนที่ตกลงไว้ ก่อนจะเอาไปเทรน"""
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame


def test_schema():
    """คอลัมน์ต้องครบ 30 ฟีเจอร์ + target และมี 569 แถว"""
    assert df.shape == (569, 31)
    assert "target" in df.columns


def test_no_missing():
    assert df.isnull().sum().sum() == 0


def test_two_classes():
    assert df["target"].nunique() == 2


def test_class_balance():
    """คลาสน้อยสุดต้องไม่ต่ำกว่า 20% (ชุดนี้ = 37.26%)"""
    assert df["target"].value_counts(normalize=True).min() >= 0.20


def test_mean_radius_range():
    """ค่าที่หลุดช่วงนี้แปลว่าข้อมูลต้นทางผิดปกติ (ค่าจริง 6.98–28.11)"""
    assert df["mean radius"].between(5.0, 30.0).all()
