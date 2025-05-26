
```bash
$ pip install -r requirements.txt

$ python train_rl.py \
     --host localhost \
     --user root \
     --password "" \
     --database mykeibadb \
     --entry_table umagoto_race_joho \
     --result_table record_master \
     --timesteps 200000 \
     --lookback_number 3 --lookback_unit MONTH

$ python plot_feature_importance.py \
  --model_path horse_racing_ppo \
  --host localhost --user root --password '' \
  --database mykeibadb --entry_table umagoto_race_joho \
  --result_table record_master --max_samples 1000

$ python plot_correlation.py \
  --host localhost --user root --password '' \
  --database mykeibadb --entry_table umagoto_race_joho \
  --result_table record_master --max_samples 5000 \
  --lookback_number 3 --lookback_unit MONTH
```


```bash
python keiba_sanshutsu/data_preprocessing/extract_features.py
python keiba_sanshutsu/data_preprocessing/clean_data.py
python keiba_sanshutsu/analysis/basic_stats.py
python keiba_sanshutsu/analysis/correlation.py
python keiba_sanshutsu/analysis/visualize.py

```