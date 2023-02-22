# handdetect
画像から手を認識する
## handcontroller.py
- 畳み込みネットワークにより手の形（指を0~5本立てている状態）をクラス分類する
## handcontrollerptn.py
- テンプレートマッチングにより画像から指先を推定し、指を何本立てているかを認識する
