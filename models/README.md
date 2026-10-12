# 保存済みモデル

`final_random_forest_pipeline.joblib`は、`02_Cancellation_Prediction.ipynb`で確定した最終Random Forest Pipelineです。欠損値補完、country上位17か国・agent上位10社への集約、One-Hot Encodingを含みます。

モデルはTrainとValidationを結合して学習し、Final Testは評価確認だけに使用しています。`src/export_final_model.py`は、同じ前処理・時系列分割・ハイパーパラメータでモデルを再作成するスクリプトです。

カスタム前処理器は`src/model_components.py`に定義しているため、Streamlit Cloudを含む公開環境でもjoblibから復元できます。
