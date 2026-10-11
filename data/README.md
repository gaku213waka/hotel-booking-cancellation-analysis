# Data directory

`hotel_bookings.csv`はGit管理対象外です。

1. [Kaggle: Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)からCSVを取得します。
2. ファイル名を`hotel_bookings.csv`として、このディレクトリに配置します。
3. `data/hotel_bookings.csv`が存在する状態でNotebookまたはSQLの読み込み処理を実行します。

本プロジェクトで確認したCSVは119,390行・32列です。先頭列のヘッダーが空欄のため、Notebookでは`hotel`へ変更しています。

原データについては、António, Almeida, and Nunes (2019), [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126)も参照してください。

