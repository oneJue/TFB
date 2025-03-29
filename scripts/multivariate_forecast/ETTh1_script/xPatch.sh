python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' --model-name "xpatch.xPatch" --model-hyper-params '{"batch_size": 128, "lr": 0.0005, "num_epochs": 20, "patience": 5, "horizon": 96,  "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/xPatch"
